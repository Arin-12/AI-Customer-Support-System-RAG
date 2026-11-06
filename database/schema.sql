CREATE TABLE IF NOT EXISTS users (
    user_id VARCHAR(20) PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(150) NOT NULL,
    phone VARCHAR(30),
    role VARCHAR(30),
    status VARCHAR(30),
    created_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS products (
    product_id VARCHAR(20) PRIMARY KEY,
    brand VARCHAR(100),
    product_name VARCHAR(200) NOT NULL,
    category VARCHAR(100),
    variant VARCHAR(100),
    price_inr NUMERIC(12, 2),
    description TEXT,
    source_name VARCHAR(200),
    source_url TEXT,
    stock_quantity INTEGER,
    status VARCHAR(30)
);

CREATE TABLE IF NOT EXISTS orders (
    order_id VARCHAR(20) PRIMARY KEY,
    user_id VARCHAR(20) NOT NULL,
    order_status VARCHAR(50),
    total_amount_inr NUMERIC(12, 2),
    tracking_number VARCHAR(100),
    expected_delivery_date DATE,
    delivery_date DATE,
    created_at TIMESTAMP,

    CONSTRAINT fk_orders_user
        FOREIGN KEY (user_id)
        REFERENCES users(user_id)
);

CREATE TABLE IF NOT EXISTS order_items (
    item_id VARCHAR(20) PRIMARY KEY,
    order_id VARCHAR(20) NOT NULL,
    product_id VARCHAR(20) NOT NULL,
    quantity INTEGER NOT NULL,
    unit_price_inr NUMERIC(12, 2),

    CONSTRAINT fk_order_items_order
        FOREIGN KEY (order_id)
        REFERENCES orders(order_id),

    CONSTRAINT fk_order_items_product
        FOREIGN KEY (product_id)
        REFERENCES products(product_id)
);

CREATE TABLE IF NOT EXISTS support_tickets (
    ticket_id VARCHAR(20) PRIMARY KEY,
    user_id VARCHAR(20) NOT NULL,
    order_id VARCHAR(20),
    category VARCHAR(100),
    priority VARCHAR(30),
    status VARCHAR(30),
    customer_message TEXT,
    ai_response TEXT,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,

    CONSTRAINT fk_tickets_user
        FOREIGN KEY (user_id)
        REFERENCES users(user_id),

    CONSTRAINT fk_tickets_order
        FOREIGN KEY (order_id)
        REFERENCES orders(order_id)
);

CREATE TABLE IF NOT EXISTS ticket_messages (
    message_id VARCHAR(20) PRIMARY KEY,
    ticket_id VARCHAR(20) NOT NULL,
    sender_type VARCHAR(30),
    message TEXT,
    timestamp TIMESTAMP,

    CONSTRAINT fk_ticket_messages_ticket
        FOREIGN KEY (ticket_id)
        REFERENCES support_tickets(ticket_id)
);