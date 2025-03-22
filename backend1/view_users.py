# from app import app, users  # Import your actual Flask app instance
# from app import User  # Import the User model

# # Ensure the Flask app context is set before querying the database
# with app.app_context():
#     users = User.query.all()
#     if users:
#         print("ID | Name | Email")
#         print("----------------------")
#         for user in users:
#             print(f"{user.id} |{user.full_name} | {user.email}")
#     else:
#         print("No users found in the database.")

from app import app, db  # Import your actual Flask app instance
from app import User  # Import the User model

# Ensure the Flask app context is set before querying the database
with app.app_context():
    users = User.query.all()
    if users:
        print("ID | Name | Email | mobileNumber   |image")
        print("----------------------")
        for user in users:
            print(f"{user.id} |{user.full_name} | {user.email} | {user.mobile_number}")
    else:
        print("No users found in the database.")
