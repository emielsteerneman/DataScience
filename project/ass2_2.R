library(readr)
library(dplyr)

data0 <- read_delim(file = "C:/My Study/A Masters/Data Science/DPV/ASS2/BI_Raw_Data - Copy.csv",
                    delim = ";", col_names = TRUE, col_types = NULL, locale = locale(encoding="ISO-8859-1"))


# make Product table 'product':
product <- data0 %>%
  select(Product_Name, Product_Category) %>%
  rename(name = Product_Name, category = Product_Category) %>%
  arrange(name, category) %>%
  group_by(name, category) %>%
  distinct() %>%
  ungroup() %>%
  mutate(productid = row_number())


# make Customer table 'customer':
customer <- data0 %>%
  select(Customer_Name, Customer_Country) %>%
  rename(name = Customer_Name, country = Customer_Country) %>%
  arrange(name, country) %>%
  group_by(name, country) %>%
  distinct() %>%
  ungroup() %>%
  mutate(customerid = row_number())

# make Sales table 'sales':
sales <- data0 %>%
  select(Order_Date_Day, Product_Order_Price_Total, Product_Name, Customer_Name) %>%
  rename(orderdate = Order_Date_Day, sales = Product_Order_Price_Total)

# Combine sales with productid and customerid
sales <- sales %>%
  full_join(product, by = c("Product_Name" = "name")) %>%
  select( -Product_Name)
sales <- sales %>%
  full_join(customer, by = c("Customer_Name" = "name")) %>%
  select( -Customer_Name)

# Remove extra columns
sales <- sales[, !(colnames(sales) %in% c("category","country"))]

# Filling the Database in PostgreSQL
library(DBI)
library(RPostgreSQL)

drv <- dbDriver("PostgreSQL")
con <- dbConnect(drv, port = 5432, host = "castle.ewi.utwente.nl",
                 dbname = "dpv2a009", user = "dpv2a009", password = "qft35ss0",
                 options="-c search_path=ass2")
dbWriteTable(con, "product", value = product, overwrite = T, row.names = F)
dbWriteTable(con, "customer", value = customer, overwrite = T, row.names = F)
dbWriteTable(con, "sales", value = sales, overwrite = T, row.names = F)
