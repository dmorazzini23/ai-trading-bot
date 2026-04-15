"""Create OMS durable event tables and baseline intent schema.

Revision ID: 20260414_0001
Revises:
Create Date: 2026-04-14 10:30:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "20260414_0001"
down_revision = None
branch_labels = None
depends_on = None


def _table_exists(table_name: str) -> bool:
    inspector = sa.inspect(op.get_bind())
    return table_name in set(inspector.get_table_names())


def _index_exists(table_name: str, index_name: str) -> bool:
    inspector = sa.inspect(op.get_bind())
    indexes = inspector.get_indexes(table_name)
    return any(str(item.get("name")) == index_name for item in indexes)


def _create_intents_table() -> None:
    if _table_exists("intents"):
        return
    op.create_table(
        "intents",
        sa.Column("intent_id", sa.String(length=128), primary_key=True),
        sa.Column("idempotency_key", sa.String(length=128), nullable=False),
        sa.Column("symbol", sa.String(length=32), nullable=False),
        sa.Column("side", sa.String(length=16), nullable=False),
        sa.Column("quantity", sa.Float(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("broker_order_id", sa.String(length=128), nullable=True),
        sa.Column("decision_ts", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.String(length=64), nullable=False),
        sa.Column("updated_at", sa.String(length=64), nullable=False),
        sa.Column(
            "submit_attempts",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column("strategy_id", sa.String(length=128), nullable=True),
        sa.Column("expected_edge_bps", sa.Float(), nullable=True),
        sa.Column("regime", sa.String(length=64), nullable=True),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("metadata_json", sa.Text(), nullable=True),
        sa.UniqueConstraint(
            "idempotency_key",
            name="uq_intents_idempotency_key",
        ),
    )
    op.create_index(
        "ix_intents_idempotency_key",
        "intents",
        ["idempotency_key"],
        unique=True,
    )
    op.create_index("ix_intents_status", "intents", ["status"], unique=False)


def _create_intent_fills_table() -> None:
    if _table_exists("intent_fills"):
        return
    op.create_table(
        "intent_fills",
        sa.Column("fill_id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("intent_id", sa.String(length=128), nullable=False),
        sa.Column("fill_qty", sa.Float(), nullable=False),
        sa.Column("fill_price", sa.Float(), nullable=True),
        sa.Column(
            "fee",
            sa.Float(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column("liquidity_flag", sa.String(length=32), nullable=True),
        sa.Column("fill_ts", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.String(length=64), nullable=False),
        sa.ForeignKeyConstraint(
            ["intent_id"],
            ["intents.intent_id"],
            name="fk_intent_fills_intent_id_intents",
        ),
    )
    op.create_index(
        "ix_intent_fills_intent_id",
        "intent_fills",
        ["intent_id"],
        unique=False,
    )


def _create_oms_events_table() -> None:
    if _table_exists("oms_events"):
        return
    op.create_table(
        "oms_events",
        sa.Column("event_id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("event_uuid", sa.String(length=128), nullable=False),
        sa.Column("intent_id", sa.String(length=128), nullable=True),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("event_ts", sa.String(length=64), nullable=False),
        sa.Column("event_source", sa.String(length=64), nullable=False),
        sa.Column("idempotency_key", sa.String(length=128), nullable=False),
        sa.Column(
            "sequence_no",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column("payload_json", sa.Text(), nullable=False),
        sa.Column("policy_hash", sa.String(length=128), nullable=True),
        sa.Column("model_hash", sa.String(length=128), nullable=True),
        sa.Column("error_code", sa.String(length=64), nullable=True),
        sa.Column("broker_order_id", sa.String(length=128), nullable=True),
        sa.Column("fill_id", sa.String(length=128), nullable=True),
        sa.Column("created_at", sa.String(length=64), nullable=False),
        sa.UniqueConstraint(
            "event_source",
            "idempotency_key",
            name="uq_oms_events_source_idempotency",
        ),
    )
    op.create_index("ix_oms_events_event_ts", "oms_events", ["event_ts"], unique=False)
    op.create_index(
        "ix_oms_events_event_type",
        "oms_events",
        ["event_type"],
        unique=False,
    )
    op.create_index(
        "ix_oms_events_intent_id",
        "oms_events",
        ["intent_id"],
        unique=False,
    )
    op.create_index(
        "ix_oms_events_event_source",
        "oms_events",
        ["event_source"],
        unique=False,
    )
    op.create_index(
        "ix_oms_events_intent_sequence",
        "oms_events",
        ["intent_id", "sequence_no"],
        unique=False,
    )
    op.create_index(
        "ix_oms_events_event_uuid",
        "oms_events",
        ["event_uuid"],
        unique=True,
    )
    op.create_index(
        "ix_oms_events_source_idempotency",
        "oms_events",
        ["event_source", "idempotency_key"],
        unique=True,
    )


def _create_decision_events_table() -> None:
    if _table_exists("decision_events"):
        return
    op.create_table(
        "decision_events",
        sa.Column("decision_id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("decision_uuid", sa.String(length=128), nullable=False),
        sa.Column("decision_ts", sa.String(length=64), nullable=False),
        sa.Column("symbol", sa.String(length=32), nullable=False),
        sa.Column("strategy_id", sa.String(length=128), nullable=True),
        sa.Column("decision_action", sa.String(length=32), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("expected_edge_bps", sa.Float(), nullable=True),
        sa.Column("policy_hash", sa.String(length=128), nullable=True),
        sa.Column("model_hash", sa.String(length=128), nullable=True),
        sa.Column("config_hash", sa.String(length=128), nullable=True),
        sa.Column("idempotency_key", sa.String(length=128), nullable=True),
        sa.Column("features_json", sa.Text(), nullable=True),
        sa.Column("context_json", sa.Text(), nullable=True),
        sa.Column("created_at", sa.String(length=64), nullable=False),
        sa.UniqueConstraint(
            "idempotency_key",
            name="uq_decision_events_idempotency_key",
        ),
    )
    op.create_index(
        "ix_decision_events_decision_ts",
        "decision_events",
        ["decision_ts"],
        unique=False,
    )
    op.create_index(
        "ix_decision_events_symbol",
        "decision_events",
        ["symbol"],
        unique=False,
    )
    op.create_index(
        "ix_decision_events_strategy_id",
        "decision_events",
        ["strategy_id"],
        unique=False,
    )
    op.create_index(
        "ix_decision_events_decision_uuid",
        "decision_events",
        ["decision_uuid"],
        unique=True,
    )
    op.create_index(
        "ix_decision_events_idempotency_key",
        "decision_events",
        ["idempotency_key"],
        unique=True,
    )


def _create_position_snapshots_table() -> None:
    if _table_exists("position_snapshots"):
        return
    op.create_table(
        "position_snapshots",
        sa.Column("snapshot_id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("snapshot_uuid", sa.String(length=128), nullable=False),
        sa.Column("snapshot_ts", sa.String(length=64), nullable=False),
        sa.Column("snapshot_source", sa.String(length=64), nullable=False),
        sa.Column("symbol", sa.String(length=32), nullable=False),
        sa.Column("quantity", sa.Float(), nullable=False),
        sa.Column("side", sa.String(length=16), nullable=True),
        sa.Column("avg_entry_price", sa.Float(), nullable=True),
        sa.Column("market_price", sa.Float(), nullable=True),
        sa.Column("market_value", sa.Float(), nullable=True),
        sa.Column("unrealized_pnl", sa.Float(), nullable=True),
        sa.Column("policy_hash", sa.String(length=128), nullable=True),
        sa.Column("model_hash", sa.String(length=128), nullable=True),
        sa.Column("idempotency_key", sa.String(length=128), nullable=False),
        sa.Column("payload_json", sa.Text(), nullable=False),
        sa.Column("created_at", sa.String(length=64), nullable=False),
        sa.UniqueConstraint(
            "snapshot_source",
            "idempotency_key",
            name="uq_position_snapshots_source_idempotency",
        ),
    )
    op.create_index(
        "ix_position_snapshots_snapshot_ts",
        "position_snapshots",
        ["snapshot_ts"],
        unique=False,
    )
    op.create_index(
        "ix_position_snapshots_snapshot_source",
        "position_snapshots",
        ["snapshot_source"],
        unique=False,
    )
    op.create_index(
        "ix_position_snapshots_symbol",
        "position_snapshots",
        ["symbol"],
        unique=False,
    )
    op.create_index(
        "ix_position_snapshots_snapshot_uuid",
        "position_snapshots",
        ["snapshot_uuid"],
        unique=True,
    )
    op.create_index(
        "ix_position_snapshots_source_idempotency",
        "position_snapshots",
        ["snapshot_source", "idempotency_key"],
        unique=True,
    )


def _create_risk_snapshots_table() -> None:
    if _table_exists("risk_snapshots"):
        return
    op.create_table(
        "risk_snapshots",
        sa.Column("risk_snapshot_id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("snapshot_uuid", sa.String(length=128), nullable=False),
        sa.Column("snapshot_ts", sa.String(length=64), nullable=False),
        sa.Column("snapshot_source", sa.String(length=64), nullable=False),
        sa.Column("idempotency_key", sa.String(length=128), nullable=False),
        sa.Column("policy_hash", sa.String(length=128), nullable=True),
        sa.Column("model_hash", sa.String(length=128), nullable=True),
        sa.Column("config_hash", sa.String(length=128), nullable=True),
        sa.Column("exposure_pct", sa.Float(), nullable=True),
        sa.Column("drawdown_pct", sa.Float(), nullable=True),
        sa.Column("var_95", sa.Float(), nullable=True),
        sa.Column("var_99", sa.Float(), nullable=True),
        sa.Column("positions_count", sa.Integer(), nullable=True),
        sa.Column("open_orders_count", sa.Integer(), nullable=True),
        sa.Column("payload_json", sa.Text(), nullable=False),
        sa.Column("created_at", sa.String(length=64), nullable=False),
        sa.UniqueConstraint(
            "snapshot_source",
            "idempotency_key",
            name="uq_risk_snapshots_source_idempotency",
        ),
    )
    op.create_index(
        "ix_risk_snapshots_snapshot_ts",
        "risk_snapshots",
        ["snapshot_ts"],
        unique=False,
    )
    op.create_index(
        "ix_risk_snapshots_snapshot_source",
        "risk_snapshots",
        ["snapshot_source"],
        unique=False,
    )
    op.create_index(
        "ix_risk_snapshots_snapshot_uuid",
        "risk_snapshots",
        ["snapshot_uuid"],
        unique=True,
    )
    op.create_index(
        "ix_risk_snapshots_source_idempotency",
        "risk_snapshots",
        ["snapshot_source", "idempotency_key"],
        unique=True,
    )


def _ensure_existing_table_indexes() -> None:
    if _table_exists("intents"):
        if not _index_exists("intents", "ix_intents_status"):
            op.create_index("ix_intents_status", "intents", ["status"], unique=False)
        if not _index_exists("intents", "ix_intents_idempotency_key"):
            op.create_index(
                "ix_intents_idempotency_key",
                "intents",
                ["idempotency_key"],
                unique=True,
            )

    if _table_exists("intent_fills") and not _index_exists(
        "intent_fills",
        "ix_intent_fills_intent_id",
    ):
        op.create_index(
            "ix_intent_fills_intent_id",
            "intent_fills",
            ["intent_id"],
            unique=False,
        )

    if _table_exists("oms_events"):
        if not _index_exists("oms_events", "ix_oms_events_event_ts"):
            op.create_index(
                "ix_oms_events_event_ts",
                "oms_events",
                ["event_ts"],
                unique=False,
            )
        if not _index_exists("oms_events", "ix_oms_events_event_type"):
            op.create_index(
                "ix_oms_events_event_type",
                "oms_events",
                ["event_type"],
                unique=False,
            )
        if not _index_exists("oms_events", "ix_oms_events_intent_id"):
            op.create_index(
                "ix_oms_events_intent_id",
                "oms_events",
                ["intent_id"],
                unique=False,
            )
        if not _index_exists("oms_events", "ix_oms_events_event_source"):
            op.create_index(
                "ix_oms_events_event_source",
                "oms_events",
                ["event_source"],
                unique=False,
            )
        if not _index_exists("oms_events", "ix_oms_events_intent_sequence"):
            op.create_index(
                "ix_oms_events_intent_sequence",
                "oms_events",
                ["intent_id", "sequence_no"],
                unique=False,
            )
        if not _index_exists("oms_events", "ix_oms_events_event_uuid"):
            op.create_index(
                "ix_oms_events_event_uuid",
                "oms_events",
                ["event_uuid"],
                unique=True,
            )
        if not _index_exists("oms_events", "ix_oms_events_source_idempotency"):
            op.create_index(
                "ix_oms_events_source_idempotency",
                "oms_events",
                ["event_source", "idempotency_key"],
                unique=True,
            )

    if _table_exists("decision_events"):
        if not _index_exists("decision_events", "ix_decision_events_decision_ts"):
            op.create_index(
                "ix_decision_events_decision_ts",
                "decision_events",
                ["decision_ts"],
                unique=False,
            )
        if not _index_exists("decision_events", "ix_decision_events_symbol"):
            op.create_index(
                "ix_decision_events_symbol",
                "decision_events",
                ["symbol"],
                unique=False,
            )
        if not _index_exists("decision_events", "ix_decision_events_strategy_id"):
            op.create_index(
                "ix_decision_events_strategy_id",
                "decision_events",
                ["strategy_id"],
                unique=False,
            )
        if not _index_exists("decision_events", "ix_decision_events_decision_uuid"):
            op.create_index(
                "ix_decision_events_decision_uuid",
                "decision_events",
                ["decision_uuid"],
                unique=True,
            )
        if not _index_exists("decision_events", "ix_decision_events_idempotency_key"):
            op.create_index(
                "ix_decision_events_idempotency_key",
                "decision_events",
                ["idempotency_key"],
                unique=True,
            )

    if _table_exists("position_snapshots"):
        if not _index_exists("position_snapshots", "ix_position_snapshots_snapshot_ts"):
            op.create_index(
                "ix_position_snapshots_snapshot_ts",
                "position_snapshots",
                ["snapshot_ts"],
                unique=False,
            )
        if not _index_exists(
            "position_snapshots",
            "ix_position_snapshots_snapshot_source",
        ):
            op.create_index(
                "ix_position_snapshots_snapshot_source",
                "position_snapshots",
                ["snapshot_source"],
                unique=False,
            )
        if not _index_exists("position_snapshots", "ix_position_snapshots_symbol"):
            op.create_index(
                "ix_position_snapshots_symbol",
                "position_snapshots",
                ["symbol"],
                unique=False,
            )
        if not _index_exists(
            "position_snapshots",
            "ix_position_snapshots_snapshot_uuid",
        ):
            op.create_index(
                "ix_position_snapshots_snapshot_uuid",
                "position_snapshots",
                ["snapshot_uuid"],
                unique=True,
            )
        if not _index_exists(
            "position_snapshots",
            "ix_position_snapshots_source_idempotency",
        ):
            op.create_index(
                "ix_position_snapshots_source_idempotency",
                "position_snapshots",
                ["snapshot_source", "idempotency_key"],
                unique=True,
            )

    if _table_exists("risk_snapshots"):
        if not _index_exists("risk_snapshots", "ix_risk_snapshots_snapshot_ts"):
            op.create_index(
                "ix_risk_snapshots_snapshot_ts",
                "risk_snapshots",
                ["snapshot_ts"],
                unique=False,
            )
        if not _index_exists("risk_snapshots", "ix_risk_snapshots_snapshot_source"):
            op.create_index(
                "ix_risk_snapshots_snapshot_source",
                "risk_snapshots",
                ["snapshot_source"],
                unique=False,
            )
        if not _index_exists("risk_snapshots", "ix_risk_snapshots_snapshot_uuid"):
            op.create_index(
                "ix_risk_snapshots_snapshot_uuid",
                "risk_snapshots",
                ["snapshot_uuid"],
                unique=True,
            )
        if not _index_exists(
            "risk_snapshots",
            "ix_risk_snapshots_source_idempotency",
        ):
            op.create_index(
                "ix_risk_snapshots_source_idempotency",
                "risk_snapshots",
                ["snapshot_source", "idempotency_key"],
                unique=True,
            )


def upgrade() -> None:
    _create_intents_table()
    _create_intent_fills_table()
    _create_oms_events_table()
    _create_decision_events_table()
    _create_position_snapshots_table()
    _create_risk_snapshots_table()
    _ensure_existing_table_indexes()


def downgrade() -> None:
    # Conservative downgrade: remove Sprint-1 event tables only.
    # Existing intent tables may already be part of live runtime state.
    if _table_exists("risk_snapshots"):
        op.drop_table("risk_snapshots")
    if _table_exists("position_snapshots"):
        op.drop_table("position_snapshots")
    if _table_exists("decision_events"):
        op.drop_table("decision_events")
    if _table_exists("oms_events"):
        op.drop_table("oms_events")
