"""evaluation audit log

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-30 14:15:39.499604
"""
import sqlalchemy as sa
from alembic import op

revision = '0002'
down_revision = '0001'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # NOT NULL on a table that may already hold rows: existing evaluations get an empty trail.
    with op.batch_alter_table('evaluations', schema=None) as batch_op:
        batch_op.add_column(sa.Column('audit_log', sa.JSON(), nullable=False, server_default=sa.text("'[]'")))


def downgrade() -> None:
    with op.batch_alter_table('evaluations', schema=None) as batch_op:
        batch_op.drop_column('audit_log')
