"""Initial PostgreSQL schema for Meridian retail data."""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "001_initial_schema"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "products",
        sa.Column("product_id", sa.String(length=32), primary_key=True),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("brand", sa.String(length=128), nullable=False),
        sa.Column("category", sa.String(length=128), nullable=False, server_default=""),
        sa.Column(
            "sub_category", sa.String(length=128), nullable=False, server_default=""
        ),
        sa.Column("price", sa.Float(), nullable=False, server_default="0"),
        sa.Column(
            "currency", sa.String(length=8), nullable=False, server_default="ZAR"
        ),
        sa.Column("stock_level", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("description", sa.Text(), nullable=False, server_default=""),
        sa.Column(
            "tags",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="[]",
        ),
    )
    op.create_index("ix_products_brand", "products", ["brand"])

    op.create_table(
        "customers",
        sa.Column("customer_id", sa.String(length=32), primary_key=True),
        sa.Column("first_name", sa.String(length=128), nullable=False),
        sa.Column("last_name", sa.String(length=128), nullable=False),
        sa.Column("email", sa.String(length=255), nullable=False, server_default=""),
        sa.Column(
            "phone_number", sa.String(length=32), nullable=False, server_default=""
        ),
        sa.Column("location", sa.String(length=255), nullable=False, server_default=""),
        sa.Column(
            "loyalty_tier",
            sa.String(length=32),
            nullable=False,
            server_default="Bronze",
        ),
        sa.Column(
            "preferred_brands",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="[]",
        ),
        sa.Column(
            "purchase_history",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="[]",
        ),
        sa.Column(
            "demographics",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="{}",
        ),
    )

    op.create_table(
        "fashion_trends",
        sa.Column("trend_id", sa.String(length=32), primary_key=True),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=False, server_default=""),
        sa.Column("season", sa.String(length=64), nullable=False, server_default=""),
        sa.Column(
            "target_demographic",
            sa.String(length=128),
            nullable=False,
            server_default="",
        ),
        sa.Column(
            "related_categories",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="[]",
        ),
        sa.Column(
            "key_colors",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="[]",
        ),
        sa.Column(
            "key_materials",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="[]",
        ),
        sa.Column("regional_relevance", sa.Text(), nullable=False, server_default=""),
    )

    op.create_table(
        "market_insights",
        sa.Column("insight_id", sa.String(length=32), primary_key=True),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("source", sa.String(length=255), nullable=False, server_default=""),
        sa.Column("date", sa.String(length=32), nullable=False, server_default=""),
        sa.Column("summary", sa.Text(), nullable=False, server_default=""),
        sa.Column(
            "data_points",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="[]",
        ),
        sa.Column(
            "regional_focus", sa.String(length=128), nullable=False, server_default=""
        ),
    )


def downgrade() -> None:
    op.drop_table("market_insights")
    op.drop_table("fashion_trends")
    op.drop_table("customers")
    op.drop_index("ix_products_brand", table_name="products")
    op.drop_table("products")
