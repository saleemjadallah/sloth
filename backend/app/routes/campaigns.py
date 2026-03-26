"""Campaign planning endpoints."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.database import get_db
from app.models.brand import Brand
from app.models.campaign import Campaign
from app.models.creative_execution import CreativeExecution
from app.schemas.campaign import CampaignResponse, CampaignSummary, CampaignUpsert
from app.services.creative_studio import CreativeStudioService

router = APIRouter(prefix="/campaigns", tags=["campaigns"])


async def _get_campaign_or_404(campaign_id: uuid.UUID, db: AsyncSession) -> Campaign:
    result = await db.execute(
        select(Campaign)
        .where(Campaign.id == campaign_id)
        .options(
            selectinload(Campaign.brand),
            selectinload(Campaign.creative_executions).selectinload(CreativeExecution.brand),
        )
    )
    campaign = result.scalar_one_or_none()
    if campaign is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Campaign {campaign_id} was not found.",
        )
    return campaign


async def _get_brand_or_404(brand_id: uuid.UUID, db: AsyncSession) -> Brand:
    result = await db.execute(select(Brand).where(Brand.id == brand_id))
    brand = result.scalar_one_or_none()
    if brand is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Brand {brand_id} was not found.",
        )
    return brand


def _serialize_campaign_summary(campaign: Campaign) -> dict[str, object]:
    return {
        "id": campaign.id,
        "brand_id": campaign.brand_id,
        "brand_name": campaign.brand.name if campaign.brand else str(campaign.brand_id),
        "brand_logo_url": campaign.brand.logo_url if campaign.brand else None,
        "name": campaign.name,
        "status": campaign.status,
        "objective": campaign.objective,
        "primary_kpi": campaign.primary_kpi,
        "start_date": campaign.start_date,
        "end_date": campaign.end_date,
        "channels": list(campaign.channels or []),
        "linked_execution_count": campaign.linked_execution_count,
        "scheduled_execution_count": campaign.scheduled_execution_count,
        "published_execution_count": campaign.published_execution_count,
        "created_at": campaign.created_at,
        "updated_at": campaign.updated_at,
    }


def _serialize_campaign_response(campaign: Campaign) -> dict[str, object]:
    payload = _serialize_campaign_summary(campaign)
    payload.update(
        {
            "audience_summary": campaign.audience_summary,
            "offer_summary": campaign.offer_summary,
            "budget_summary": campaign.budget_summary,
            "cadence_summary": campaign.cadence_summary,
            "messaging_pillars": list(campaign.messaging_pillars or []),
            "notes": campaign.notes,
            "brand": {
                "id": campaign.brand.id,
                "name": campaign.brand.name,
                "website_url": campaign.brand.website_url,
                "logo_url": campaign.brand.logo_url,
                "industry": campaign.brand.industry,
            },
            "executions": [
                CreativeStudioService.serialize_saved_execution_summary(record)
                for record in sorted(
                    campaign.creative_executions,
                    key=lambda record: record.updated_at or record.created_at,
                    reverse=True,
                )
            ],
        }
    )
    return payload


async def _sync_campaign_executions(
    campaign: Campaign,
    execution_ids: list[uuid.UUID],
    db: AsyncSession,
) -> None:
    selected_ids = list(dict.fromkeys(execution_ids))

    current_result = await db.execute(
        select(CreativeExecution).where(CreativeExecution.campaign_id == campaign.id)
    )
    current_records = current_result.scalars().all()

    if not selected_ids:
        for record in current_records:
            record.campaign_id = None
        return

    selected_result = await db.execute(
        select(CreativeExecution).where(
            CreativeExecution.brand_id == campaign.brand_id,
            CreativeExecution.id.in_(selected_ids),
        )
    )
    selected_records = selected_result.scalars().all()
    selected_by_id = {record.id: record for record in selected_records}

    missing_ids = [execution_id for execution_id in selected_ids if execution_id not in selected_by_id]
    if missing_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="One or more execution_ids do not belong to the selected brand.",
        )

    conflicting_records = [
        record
        for record in selected_records
        if record.campaign_id is not None and record.campaign_id != campaign.id
    ]
    if conflicting_records:
        names = ", ".join(record.concept_name for record in conflicting_records[:3])
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Selected executions are already linked to another campaign: {names}",
        )

    selected_id_set = set(selected_ids)
    for record in current_records:
        if record.id not in selected_id_set:
            record.campaign_id = None

    for execution_id in selected_ids:
        selected_by_id[execution_id].campaign_id = campaign.id


@router.get("", response_model=list[CampaignSummary], summary="List campaigns")
async def list_campaigns(
    brand_id: uuid.UUID | None = Query(default=None),
    db: AsyncSession = Depends(get_db),
) -> list[dict[str, object]]:
    query = (
        select(Campaign)
        .options(
            selectinload(Campaign.brand),
            selectinload(Campaign.creative_executions),
        )
        .order_by(Campaign.updated_at.desc(), Campaign.created_at.desc())
    )
    if brand_id is not None:
        query = query.where(Campaign.brand_id == brand_id)

    result = await db.execute(query)
    campaigns = result.scalars().all()
    return [_serialize_campaign_summary(campaign) for campaign in campaigns]


@router.post(
    "",
    response_model=CampaignResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a campaign",
)
async def create_campaign(
    body: CampaignUpsert,
    db: AsyncSession = Depends(get_db),
) -> dict[str, object]:
    await _get_brand_or_404(body.brand_id, db)
    campaign = Campaign(
        brand_id=body.brand_id,
        name=body.name.strip(),
        status=body.status.strip() or "draft",
        objective=body.objective.strip(),
        audience_summary=body.audience_summary.strip(),
        offer_summary=body.offer_summary.strip(),
        primary_kpi=body.primary_kpi.strip(),
        budget_summary=body.budget_summary.strip() if body.budget_summary else None,
        start_date=body.start_date,
        end_date=body.end_date,
        cadence_summary=body.cadence_summary.strip() if body.cadence_summary else None,
        channels=body.channels,
        messaging_pillars=body.messaging_pillars,
        notes=body.notes.strip() if body.notes else None,
    )
    db.add(campaign)
    await db.flush()
    await _sync_campaign_executions(campaign, body.execution_ids, db)
    await db.flush()
    campaign = await _get_campaign_or_404(campaign.id, db)
    return _serialize_campaign_response(campaign)


@router.get(
    "/{campaign_id}",
    response_model=CampaignResponse,
    summary="Get a campaign",
)
async def get_campaign(
    campaign_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> dict[str, object]:
    campaign = await _get_campaign_or_404(campaign_id, db)
    return _serialize_campaign_response(campaign)


@router.put(
    "/{campaign_id}",
    response_model=CampaignResponse,
    summary="Update a campaign",
)
async def update_campaign(
    campaign_id: uuid.UUID,
    body: CampaignUpsert,
    db: AsyncSession = Depends(get_db),
) -> dict[str, object]:
    campaign = await _get_campaign_or_404(campaign_id, db)
    await _get_brand_or_404(body.brand_id, db)

    if campaign.brand_id != body.brand_id:
        current_result = await db.execute(
            select(CreativeExecution).where(CreativeExecution.campaign_id == campaign.id)
        )
        for record in current_result.scalars().all():
            record.campaign_id = None

    campaign.brand_id = body.brand_id
    campaign.name = body.name.strip()
    campaign.status = body.status.strip() or "draft"
    campaign.objective = body.objective.strip()
    campaign.audience_summary = body.audience_summary.strip()
    campaign.offer_summary = body.offer_summary.strip()
    campaign.primary_kpi = body.primary_kpi.strip()
    campaign.budget_summary = body.budget_summary.strip() if body.budget_summary else None
    campaign.start_date = body.start_date
    campaign.end_date = body.end_date
    campaign.cadence_summary = body.cadence_summary.strip() if body.cadence_summary else None
    campaign.channels = body.channels
    campaign.messaging_pillars = body.messaging_pillars
    campaign.notes = body.notes.strip() if body.notes else None

    await _sync_campaign_executions(campaign, body.execution_ids, db)
    await db.flush()
    campaign = await _get_campaign_or_404(campaign.id, db)
    return _serialize_campaign_response(campaign)


@router.delete(
    "/{campaign_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_model=None,
    summary="Delete a campaign",
)
async def delete_campaign(
    campaign_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> None:
    campaign = await _get_campaign_or_404(campaign_id, db)
    current_result = await db.execute(
        select(CreativeExecution).where(CreativeExecution.campaign_id == campaign.id)
    )
    for record in current_result.scalars().all():
        record.campaign_id = None
    await db.delete(campaign)
    await db.flush()
