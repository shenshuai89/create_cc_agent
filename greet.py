import sys
from typing import Optional
def greet(name: str = "World") -> None:
    """
    Greet the user with a personalized message.
    
    Args:
        name (str): The name of the person to greet. Defaults to "World".
    """
    print(f"Hello, {name}!")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        greet(sys.argv[1])
    else:
        greet()