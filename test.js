THIS IS 4TH single symmetric key

from cryptography.fernet import Fernet

key = Fernet.generate_key()
key

crypter = Fernet(key)
pw = crypter.encrypt(b'Mypassword')

pw

str(pw, 'utf-8')

decryptString = crypter.decrypt(pw)

str(pw, 'utf8')

str(decryptString, 'utf8')





5th asymmetric 

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.exceptions import InvalidSignature

def generate_key_pair():
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend()
    )
    public_key = private_key.public_key()
    return private_key, public_key

def sign_document(private_key, document):
    signature = private_key.sign(
        document,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH
        ),
        hashes.SHA256()
    )
    return signature

def verify_signature(public_key, document, signature):
    try:
        public_key.verify(
            signature,
            document,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return True
    except InvalidSignature:
        return False

# Example usage:
if __name__ == "__main__":
    # Generate key pair
    private_key, public_key = generate_key_pair()

    # Document to be signed
    document = b"This is the document to be signed."

    # Sign the document
    signature = sign_document(private_key, document)

    # Verify the signature
    if verify_signature(public_key, document, signature):
        print("Signature is valid.")
    else:
        print("Signature is invalid.")



6th.  caeser cipher

def caesar_encrypt(plain_text, shift):
    encrypted_text = ""
    for char in plain_text:
        if char.isalpha():
            shifted = ord(char) + shift
            if char.islower():
                if shifted > ord('z'):
                    shifted -= 26
                elif shifted < ord('a'):
                    shifted += 26
            elif char.isupper():
                if shifted > ord('Z'):
                    shifted -= 26
                elif shifted < ord('A'):
                    shifted += 26
            encrypted_text += chr(shifted)
        else:
            encrypted_text += char
    return encrypted_text

def caesar_decrypt(encrypted_text, shift):
    decrypted_text = ""
    for char in encrypted_text:
        if char.isalpha():
            shifted = ord(char) - shift
            if char.islower():
                if shifted > ord('z'):
                    shifted -= 26
                elif shifted < ord('a'):
                    shifted += 26
            elif char.isupper():
                if shifted > ord('Z'):
                    shifted -= 26
                elif shifted < ord('A'):
                    shifted += 26
            decrypted_text += chr(shifted)
        else:
            decrypted_text += char
    return decrypted_text

# Example usage
text = "Hello, World!"
shift = 3

encrypted_text = caesar_encrypt(text, shift)
print("Encrypted:", encrypted_text)

decrypted_text = caesar_decrypt(encrypted_text, shift)
print("Decrypted:", decrypted_text)




7th.  substitution cipher


import random

def generate_cipher_key():
    # Generate a random permutation of uppercase letters
    alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    random.shuffle(alphabet)
    return dict(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", alphabet))

def substitution_encrypt(plain_text, cipher_key):
    encrypted_text = ""
    for char in plain_text:
        if char.isalpha() and char.isupper():
            encrypted_text += cipher_key[char]
        else:
            encrypted_text += char
    return encrypted_text

def substitution_decrypt(encrypted_text, cipher_key):
    decrypted_text = ""
    for char in encrypted_text:
        if char.isalpha() and char.isupper():
            for key, value in cipher_key.items():
                if value == char:
                    decrypted_text += key
                    break
        else:
            decrypted_text += char
    return decrypted_text

# Example usage
plain_text = "HELLO WORLD"
cipher_key = generate_cipher_key()

print("Original:", plain_text)
encrypted_text = substitution_encrypt(plain_text, cipher_key)
print("Encrypted:", encrypted_text)
decrypted_text = substitution_decrypt(encrypted_text, cipher_key)
print("Decrypted:", decrypted_text)





8th.   aes

from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
import os
import secrets

def pad(s):
    return s + b"\0" * (AES.block_size - len(s) % AES.block_size)

def encrypt_message(key, message):
    message = pad(message)
    iv = get_random_bytes(AES.block_size)
    cipher = AES.new(key, AES.MODE_CBC, iv)
    return iv + cipher.encrypt(message)

def decrypt_message(key, ciphertext):
    iv = ciphertext[:AES.block_size]
    cipher = AES.new(key, AES.MODE_CBC, iv)
    plaintext = cipher.decrypt(ciphertext[AES.block_size:])
    return plaintext.rstrip(b"\0")

def encrypt_file(file_path, key):
    with open(file_path, 'rb') as f:
        plaintext = f.read()
    ciphertext = encrypt_message(key, plaintext)
    with open(file_path + '.enc', 'wb') as f:
        f.write(ciphertext)
    os.remove(file_path)

def decrypt_file(file_path, key):
    with open(file_path, 'rb') as f:
        ciphertext = f.read()
    plaintext = decrypt_message(key, ciphertext)
    with open(os.path.splitext(file_path)[0], 'wb') as f:
        f.write(plaintext)
    os.remove(file_path)

def encrypt_folder(folder_path, key):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            encrypt_file(file_path, key)

def decrypt_folder(folder_path, key):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.enc'):
                file_path = os.path.join(root, file)
                decrypt_file(file_path, key)

# Generate a random key of 16 bytes
key = secrets.token_bytes(16)

folder_path = "C:\\Users\\Dell\\Desktop\\final mst\\exp8"

# Encrypt files in the folder
#encrypt_folder(folder_path, key)

# Decrypt files in the folder
decrypt_folder(folder_path, key)





9th rsa
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP

def generate_key_pair():
    key = RSA.generate(2048)
    private_key = key.export_key()
    public_key = key.publickey().export_key()
    return private_key, public_key

def encrypt_data(public_key, data):
    key = RSA.import_key(public_key)
    cipher_rsa = PKCS1_OAEP.new(key)
    encrypted_data = cipher_rsa.encrypt(data.encode())
    return encrypted_data

def decrypt_data(private_key, encrypted_data):
    key = RSA.import_key(private_key)
    cipher_rsa = PKCS1_OAEP.new(key)
    decrypted_data = cipher_rsa.decrypt(encrypted_data)
    return decrypted_data.decode()

# Example usage
if __name__ == "__main__":
    # Generate key pair
    private_key, public_key = generate_key_pair()

    # Encrypt data
    data = "Hello, Today is my Labmst."
    encrypted_data = encrypt_data(public_key, data)
    print("Encrypted data:", encrypted_data)

    # Decrypt data
    decrypted_data = decrypt_data(private_key, encrypted_data)
    print("Decrypted data:", decrypted_data)




10th. diffie


import random

def generate_prime():
    primes = [i for i in range(2, 100) if all(i % j != 0 for j in range(2, i))]
    return random.choice(primes)

def generate_public_private_key(prime):
    private_key = random.randint(2, prime - 1)
    public_key = pow(2, private_key) % prime
    return private_key, public_key

def generate_shared_secret(private_key, other_public_key, prime):
    shared_secret = pow(other_public_key, private_key) % prime
    return shared_secret

def encrypt(message, shared_secret):
    encrypted_message = ""
    for char in message:
        encrypted_char = chr(ord(char) + shared_secret)
        encrypted_message += encrypted_char
    return encrypted_message

def decrypt(encrypted_message, shared_secret):
    decrypted_message = ""
    for char in encrypted_message:
        decrypted_char = chr(ord(char) - shared_secret)
        decrypted_message += decrypted_char
    return decrypted_message

def main():
    prime = generate_prime()
    print("Prime number generated:", prime)

    alice_private_key, alice_public_key = generate_public_private_key(prime)
    bob_private_key, bob_public_key = generate_public_private_key(prime)

    print("Alice's private key:", alice_private_key)
    print("Alice's public key:", alice_public_key)
    print("Bob's private key:", bob_private_key)
    print("Bob's public key:", bob_public_key)

    alice_shared_secret = generate_shared_secret(alice_private_key, bob_public_key, prime)
    bob_shared_secret = generate_shared_secret(bob_private_key, alice_public_key, prime)

    print("Shared secret computed by Alice:", alice_shared_secret)
    print("Shared secret computed by Bob:", bob_shared_secret)

    message = "Hello, Bob! This is a secret message."
    print("Original message:", message)

    encrypted_message = encrypt(message, alice_shared_secret)
    print("Encrypted message:", encrypted_message)

    decrypted_message = decrypt(encrypted_message, bob_shared_secret)
    print("Decrypted message:", decrypted_message)

if __name__ == "__main__":
    main()








