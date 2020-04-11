# The if __name__ == '__main__': protects the full_code from being run in another file

# Define printing functions
def full_code():
    print('This is being run directly and in reality would be doing lots of things')

def free_code():
    print('This will always be run')


free_code()    
if __name__ == '__main__':
    full_code()
else:
    print('This is being imported by another code')

