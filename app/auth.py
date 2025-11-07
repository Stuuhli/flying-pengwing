"""Create user creates a password and username storing it in db, and then authentication is done before session is initialized. Password is hashed."""
import bcrypt


""" Initialize the hashing algo """
salt = bcrypt.gensalt()


def password_create(password: str):
    """ Create password using bcrypt 
    """
    bytes = password.encode('utf-8') 
    return bcrypt.hashpw(bytes, salt=salt)

def password_verify(password: str, hashed: str):
    """ Compare the entered password with the hashed password 
    """
    bytes = password.encode('utf-8') 
    hashed = hashed.encode('utf-8')
    if bcrypt.checkpw(bytes,hashed):
        return True
    else:
        return False
