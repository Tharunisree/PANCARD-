from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.declarative import declarative_base
import urllib.parse

# Encode the password properly
encoded_password = urllib.parse.quote_plus('root')

# Update the URL_DATABASE with the encoded password
URL_DATABASE = f'mysql+pymysql://root:{encoded_password}@localhost:3306/signature'
engine = create_engine(URL_DATABASE)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()
