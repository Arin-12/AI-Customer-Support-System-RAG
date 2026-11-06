import pandas as pd
from pathlib import Path


PRODUCT_FILE = Path("data/knowledge_base/product_information.csv")


def load_products():
    """Load the product information CSV."""
    return pd.read_csv(PRODUCT_FILE)


def test_product_file_exists():
    assert PRODUCT_FILE.exists(), (
        f"Product information file not found: {PRODUCT_FILE}"
    )


def test_product_file_not_empty():
    products = load_products()

    assert not products.empty
    assert len(products) > 0


def test_required_columns_exist():
    products = load_products()

    required_columns = [
        "product_id",
        "brand",
        "product_name",
        "category",
        "variant",
        "description",
        "reference_price_inr",
        "features",
        "care",
        "warranty_note",
        "source_name",
        "source_url",
    ]

    for column in required_columns:
        assert column in products.columns


def test_product_ids_are_unique():
    products = load_products()

    assert products["product_id"].notna().all()
    assert products["product_id"].is_unique


def test_expected_products_exist():
    products = load_products()

    expected_products = {
        "PROD001": "iPhone 16",
        "PROD002": "iPhone 16 Plus",
        "PROD003": "iPhone 16 Pro",
        "PROD004": "iPhone 16 Pro Max",
    }

    for product_id, product_name in expected_products.items():
        matching = products[products["product_id"] == product_id]

        assert len(matching) == 1
        assert matching.iloc[0]["product_name"] == product_name


def test_product_brands():
    products = load_products()

    apple_products = products[
        products["product_id"].isin(
            ["PROD001", "PROD002", "PROD003", "PROD004"]
        )
    ]

    assert len(apple_products) == 4
    assert (apple_products["brand"] == "Apple").all()


def test_product_categories():
    products = load_products()

    iphone_products = products[
        products["product_id"].isin(
            ["PROD001", "PROD002", "PROD003", "PROD004"]
        )
    ]

    assert (iphone_products["category"] == "Smartphone").all()


def test_product_prices_are_valid():
    products = load_products()

    iphone_products = products[
        products["product_id"].isin(
            ["PROD001", "PROD002", "PROD003", "PROD004"]
        )
    ]

    assert pd.api.types.is_numeric_dtype(
        iphone_products["reference_price_inr"]
    )

    assert (iphone_products["reference_price_inr"] > 0).all()


def test_product_source_information():
    products = load_products()

    iphone_products = products[
        products["product_id"].isin(
            ["PROD001", "PROD002", "PROD003", "PROD004"]
        )
    ]

    assert iphone_products["source_name"].notna().all()
    assert iphone_products["source_url"].notna().all()

    for url in iphone_products["source_url"]:
        assert str(url).startswith("https://")


def test_product_information_complete():
    products = load_products()

    iphone_products = products[
        products["product_id"].isin(
            ["PROD001", "PROD002", "PROD003", "PROD004"]
        )
    ]

    important_columns = [
        "product_id",
        "brand",
        "product_name",
        "category",
        "variant",
        "description",
        "features",
        "care",
        "warranty_note",
        "source_name",
        "source_url",
    ]

    for column in important_columns:
        assert iphone_products[column].notna().all()
        assert (
            iphone_products[column]
            .astype(str)
            .str.strip()
            .ne("")
            .all()
        )