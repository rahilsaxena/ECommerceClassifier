from class_a import A  # Import Class A from class_a.py

class B:
    def call_a_method(self):
        a = A()  # Create an instance of Class A
        a.method_in_a()  # Call the method from Class A

# Optional usage if you want to test this file directly
if __name__ == "__main__":
    b = B()
    b.call_a_method()