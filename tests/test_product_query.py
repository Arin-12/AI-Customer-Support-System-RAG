import pandas as pd
from pathlib import Path


PRODUCT_FILE = Path("data/knowledge_base/product_information.csv")


def load_products():
    return pd.read_csv(PRODUCT_FILE)


def find_product(query: str):
    """
    Find a product by product ID, product name, brand,
    or partial product name.
    """

    products = load_products()

    query = query.strip().lower()

    # Exact product ID
    exact_id = products[
        products["product_id"].astype(str).str.lower() == query
    ]

    if not exact_id.empty:
        return exact_id.iloc[0].to_dict()

    # Exact product name
    exact_name = products[
        products["product_name"].astype(str).str.lower() == query
    ]

    if not exact_name.empty:
        return exact_name.iloc[0].to_dict()

    # Partial product name
    matches = products[
        products["product_name"]
        .astype(str)
        .str.lower()
        .str.contains(query, na=False)
    ]

    if not matches.empty:
        return matches.iloc[0].to_dict()

    # Brand search
    brand_matches = products[
        products["brand"]
        .astype(str)
        .str.lower()
        .str.contains(query, na=False)
    ]

    if not brand_matches.empty:
        return brand_matches.iloc[0].to_dict()

    return None


def format_product(product: dict) -> str:
    """Format product information for display."""

    return f"""
Product ID: {product["product_id"]}
Brand: {product["brand"]}
Product Name: {product["product_name"]}
Category: {product["category"]}
Variant: {product["variant"]}
Description: {product["description"]}
Reference Price: ₹{product["reference_price_inr"]:,.0f}
Features: {product["features"]}
Care: {product["care"]}
Warranty: {product["warranty_note"]}
Source: {product["source_name"]}
Source URL: {product["source_url"]}
""".strip()


def ask_product():
    print("=" * 70)
    print("PRODUCT INFORMATION SYSTEM")
    print("=" * 70)
    print("Ask using:")
    print("- Product ID  (e.g. PROD003)")
    print("- Product name (e.g. iPhone 16 Pro)")
    print("- Brand (e.g. Apple)")
    print("- Partial name (e.g. iPhone 16)")
    print("Type 'exit' to quit.")

    while True:
        query = input("\nEnter product: ").strip()

        if query.lower() == "exit":
            print("Goodbye!")
            break

        if not query:
            print("Please enter a product name or product ID.")
            continue

        product = find_product(query)

        if product is None:
            print("\nProduct not found.")
            continue

        print("\n" + "=" * 70)
        print("PRODUCT INFORMATION")
        print("=" * 70)
        print(format_product(product))


if __name__ == "__main__":
    ask_product()