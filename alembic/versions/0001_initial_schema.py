"""initial schema

Revision ID: 0001
Revises: 
Create Date: 2026-09-30 13:29:46.400928
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = '0001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table('disputes',
    sa.Column('case_id', sa.String(length=64), nullable=False),
    sa.Column('merchant_id', sa.String(length=32), nullable=False),
    sa.Column('reason_code', sa.String(length=16), nullable=False),
    sa.Column('category', sa.String(length=32), nullable=False),
    sa.Column('amount_inr', sa.Numeric(precision=12, scale=2), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.PrimaryKeyConstraint('case_id')
    )
    with op.batch_alter_table('disputes', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_disputes_merchant_id'), ['merchant_id'], unique=False)

    op.create_table('evaluations',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('case_id', sa.String(length=64), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('input_hash', sa.String(length=64), nullable=False),
    sa.Column('action', sa.Enum('CONTEST', 'CONCEDE', 'ESCALATE', name='action_enum'), nullable=False),
    sa.Column('confidence', sa.String(length=8), nullable=False),
    sa.Column('p_win', sa.Float(), nullable=False),
    sa.Column('p_win_std', sa.Float(), nullable=False),
    sa.Column('ev_contest_inr', sa.Numeric(precision=12, scale=2), nullable=False),
    sa.Column('contest_cost_inr', sa.Numeric(precision=12, scale=2), nullable=False),
    sa.Column('review_value_inr', sa.Numeric(precision=12, scale=2), nullable=False),
    sa.Column('review_cost_inr', sa.Numeric(precision=12, scale=2), nullable=False),
    sa.Column('facts', sa.JSON(), nullable=False),
    sa.Column('fact_agreement', sa.JSON(), nullable=False),
    sa.Column('unread_facts', sa.JSON(), nullable=False),
    sa.Column('conflicts', sa.JSON(), nullable=False),
    sa.Column('reasons', sa.JSON(), nullable=False),
    sa.Column('counterfactual', sa.JSON(), nullable=False),
    sa.Column('contributions', sa.JSON(), nullable=False),
    sa.Column('extraction_source', sa.String(length=24), nullable=False),
    sa.Column('degraded_reason', sa.String(length=32), nullable=True),
    sa.Column('llm_calls', sa.Integer(), nullable=False),
    sa.Column('llm_tokens', sa.Integer(), nullable=False),
    sa.Column('model_version', sa.String(length=32), nullable=False),
    sa.Column('latency_ms', sa.Integer(), nullable=False),
    sa.Column('input_snapshot', sa.JSON(), nullable=False),
    sa.Column('merchant_snapshot', sa.JSON(), nullable=False),
    sa.ForeignKeyConstraint(['case_id'], ['disputes.case_id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('evaluations', schema=None) as batch_op:
        batch_op.create_index('ix_evaluations_case_id_id', ['case_id', 'id'], unique=False)
        batch_op.create_index(batch_op.f('ix_evaluations_input_hash'), ['input_hash'], unique=False)

    op.create_table('outcomes',
    sa.Column('case_id', sa.String(length=64), nullable=False),
    sa.Column('outcome', sa.Enum('won', 'lost', name='outcome_enum'), nullable=False),
    sa.Column('recorded_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['case_id'], ['disputes.case_id'], ),
    sa.PrimaryKeyConstraint('case_id')
    )
    op.create_table('idempotency_keys',
    sa.Column('key', sa.String(length=128), nullable=False),
    sa.Column('request_hash', sa.String(length=64), nullable=False),
    sa.Column('evaluation_id', sa.Integer(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['evaluation_id'], ['evaluations.id'], ),
    sa.PrimaryKeyConstraint('key'),
    sa.UniqueConstraint('key', name='uq_idempotency_key')
    )
    op.create_table('reviews',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('evaluation_id', sa.Integer(), nullable=False),
    sa.Column('case_id', sa.String(length=64), nullable=False),
    sa.Column('reviewer', sa.String(length=64), nullable=False),
    # action_enum already exists (created with 'evaluations'); do not CREATE TYPE again on Postgres.
    sa.Column('action', postgresql.ENUM('CONTEST', 'CONCEDE', 'ESCALATE', name='action_enum', create_type=False), nullable=False),
    sa.Column('note', sa.Text(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['case_id'], ['disputes.case_id'], ),
    sa.ForeignKeyConstraint(['evaluation_id'], ['evaluations.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('evaluation_id')
    )
    with op.batch_alter_table('reviews', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_reviews_case_id'), ['case_id'], unique=False)



def downgrade() -> None:
    with op.batch_alter_table('reviews', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_reviews_case_id'))

    op.drop_table('reviews')
    op.drop_table('idempotency_keys')
    op.drop_table('outcomes')
    with op.batch_alter_table('evaluations', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_evaluations_input_hash'))
        batch_op.drop_index('ix_evaluations_case_id_id')

    op.drop_table('evaluations')
    with op.batch_alter_table('disputes', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_disputes_merchant_id'))

    op.drop_table('disputes')
    sa.Enum(name='outcome_enum').drop(op.get_bind(), checkfirst=True)
    sa.Enum(name='action_enum').drop(op.get_bind(), checkfirst=True)
