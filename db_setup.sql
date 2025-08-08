-- Create a new database for the application
CREATE DATABASE IF NOT EXISTS flight_app_db;

-- Use the newly created database
USE flight_app_db;

-- Create a table to store user credentials
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) NOT NULL UNIQUE,
    password VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);