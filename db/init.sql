CREATE DATABASE challenge;
use challenge;

CREATE TABLE sales (
  SALE_ID INT AUTO_INCREMENT PRIMARY KEY,
  PROD_ID VARCHAR(10) NOT NULL,
  DATE_ORDER DATE NOT NULL,
  QTY_ORDER DECIMAL(6,2) NOT NULL,
  REVENUE DECIMAL(6,2) NOT NULL
);

CREATE TABLE comp_prices (
  COMP_ID INT AUTO_INCREMENT PRIMARY KEY,
  PROD_ID VARCHAR(10) NOT NULL,
  DATE_EXTRACTION DATE NOT NULL,
  COMPETITOR VARCHAR(10) NOT NULL,
  COMPETITOR_PRICE DECIMAL(6,2) NOT NULL,
  PAY_TYPE INTEGER NOT NULL
);

LOAD DATA LOCAL INFILE '/var/lib/mysql-files/sales.csv' 
INTO TABLE sales 
FIELDS TERMINATED BY ',' 
LINES TERMINATED BY '\n'
IGNORE 1 LINES
(PROD_ID, DATE_ORDER, QTY_ORDER, REVENUE);  

LOAD DATA LOCAL INFILE '/var/lib/mysql-files/comp_prices.csv' 
INTO TABLE comp_prices 
FIELDS TERMINATED BY ',' 
LINES TERMINATED BY '\n'
IGNORE 1 LINES
(PROD_ID,DATE_EXTRACTION,COMPETITOR,COMPETITOR_PRICE,PAY_TYPE); 