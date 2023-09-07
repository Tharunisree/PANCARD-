# from sqlalchemy import column,Boolean, Integer, String,table

# # from db import Base
# from sqlalchemy import Column, Integer, String
# from db import Base

# class User(Base):
#     __tablename__ = 'col_extraction'

#     id = Column(Integer, primary_key=True, index=True)
#     c_code = Column(String(50))
#     percentage = Column(String(50))


# from sqlalchemy import Column, Integer, LargeBinary, String, Float
# from db import Base

# class ColorEntry(Base):
#     __tablename__ = 'color_entries'

#     id = Column(Integer, primary_key=True, index=True)
#     image = Column(LargeBinary)
#     c_code = Column(String(50))
#     percentage = Column(Float)

# from sqlalchemy import Column, Integer, LargeBinary, Text
# from db import Base

# # class ColorEntry(Base):
# #     __tablename__ = 'color_entries_1'

# #     id = Column(Integer, primary_key=True, index=True)
# #     image = Column(LargeBinary)
# #     color_data = Column(Text)  # Store color codes and percentages in JSON format


# from sqlalchemy import Column, Integer, LargeBinary, String
# from db import Base

# class ColorEntry1(Base):
#     __tablename__ = 'tenkai'

#     id = Column(Integer, primary_key=True, index=True)
#     image = Column(LargeBinary)  # Binary data for the image
#     color_data = Column(String(1000))   # JSON-formatted color data

# class CapturedImage(Base):
#     __tablename__ = "Chennakai"

#     id = Column(Integer, primary_key=True, index=True)
#     image_data = Column(LargeBinary)

    



from sqlalchemy import column,Boolean, Integer, String,table

# from db import Base
from sqlalchemy import Column, Integer, String
from db import Base

class User(Base):
    __tablename__ = 'col_extraction'

    id = Column(Integer, primary_key=True, index=True)
    c_code = Column(String(50))
    percentage = Column(String(50))


from sqlalchemy import Column, Integer, LargeBinary, String, Float
from db import Base

class ColorEntry(Base):
    __tablename__ = 'color_entries'

    id = Column(Integer, primary_key=True, index=True)
    image = Column(LargeBinary)
    c_code = Column(String(50))
    percentage = Column(Float)


from sqlalchemy import Column, Integer, LargeBinary, Text
from db import Base

# class ColorEntry(Base):
#     _tablename_ = 'color_entries_1'

#     id = Column(Integer, primary_key=True, index=True)
#     image = Column(LargeBinary)
#     color_data = Column(Text)  # Store color codes and percentages in JSON format


from sqlalchemy import Column, Integer, LargeBinary, String
from db import Base

from sqlalchemy import Column, Integer, String, LargeBinary
from sqlalchemy.orm import declarative_base
Base = declarative_base()

class ColorEntry1(Base):
    __tablename__ = 'color_entries_1'

    id = Column(Integer, primary_key=True, index=True)
    image = Column(LargeBinary)  # Binary data for the image
    color_data = Column(String(1000))   # JSON-formatted color data


## kiran 1

# class CapturedImage(Base):
#     __tablename__ = "captured_images"

#     id = Column(Integer, primary_key=True, index=True)
#     image_data = Column(LargeBinary)



class CapturedImage(Base):
    __tablename__ = "face_detects"

    id = Column(Integer, primary_key=True, index=True)
    image_data = Column(LargeBinary)



# from sqlalchemy import Column, Integer, String, LargeBinary
# from sqlalchemy.orm import declarative_base


# class PancardDetail(Base):
#     __tablename__ = 'pancard_details'

#     id = Column(Integer, primary_key=True, index=True)
#     image = Column(LargeBinary)
#     color_data = Column(String(1000))
#     face_image = Column(LargeBinary)


class ImageEntry1(Base):
    __tablename__ = "api"

    id = Column(Integer, primary_key=True, index=True)
    thumb = Column(LargeBinary)  # Binary data for the image


class CapturedImage2(Base):
    __tablename__ = "pattern_detects"

    id = Column(Integer, primary_key=True, index=True)
    image_data = Column(LargeBinary)


# class CapturedImage3(Base):
#     __tablename__ = "pattern_detect"

#     id = Column(Integer, primary_key=True, index=True)
#     image_data = Column(LargeBinary)



# class Text(Base):
#     __tablename__ = "text_extraction"

#     id = Column(Integer, primary_key=True, index=True)
#     text = Column(String(10000))

# from pydantic import Base

class AnalysisResult(Base):
    __tablename__ = "analysis_results"

    id = Column(Integer, primary_key=True, index=True)
    text = Column(String(10000))





# Base = declarative_base()

class Image(Base):
    __tablename__ = "images"

    id = Column(Integer, primary_key=True, index=True)
    image = Column(LargeBinary)
