import bcrypt
from base.com.vo.detection_vo import AdminVO # Import your existing User model
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import time
# Function to create a new user (admin)
# def create_admin(session, username, password):
#     hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
#     admin = AdminVO(login_username=username, login_password=hashed_password)
#     session.add(admin)
#     session.commit()
#     print("Admin created successfully!")

from datetime import datetime

def create_admin(session, username, password):
    # Hash the provided password
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    
    # Get the current epoch time
    current_epoch_time = int(time.time())
    
    # Create a new admin instance with the provided username, hashed password, and current epoch time
    admin = AdminVO(login_username=username, login_password=hashed_password, created_on=current_epoch_time, modified_on=current_epoch_time)
    
    # Add the new admin instance to the session
    session.add(admin)
    
    # Commit the session to save the admin to the database
    session.commit()
    
    print("Admin created successfully!")


def verify_login(session, username, password):
    user = session.query(AdminVO).filter_by(login_username=username).first()
    if user:
        hashed_password = user.login_password
        if bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8')):
            print("Login successful!")
            # Redirect user to another page or proceed with successful login logic here
        else:
            print("Incorrect password! Please try again.")
            # Allow the user to remain on the same page without redirection
    else:
        print("User not found! Please try again.")
        # Allow the user to remain on the same page without redirection



def main():
    # Connect to MySQL database
    engine = create_engine('mysql+pymysql://root:root@localhost:3306/dashboard?charset=latin1')
    Session = sessionmaker(bind=engine)
    session = Session()

    # Prompt admin creation
    create_admin_input = input("Do you want to create an admin account? (yes/no): ")
    if create_admin_input.lower() == 'yes':
        admin_username = input("Enter admin username: ")
        admin_password = input("Enter admin password: ")
        create_admin(session, admin_username, admin_password)
        # Commit changes to the database before verifying login
        session.commit()

    # Prompt for login
    username = input("Enter your username: ")
    password = input("Enter your password: ")
    verify_login(session, username, password)

    # Close session
    session.close()

if __name__ == "__main__":
    main()
