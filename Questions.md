# Questions and Answers

### 1. Can we add elements of different data types in the same list?
**Answer**: Yes, Python lists can hold elements of different data types. For example, you can have a list that contains integers, strings, and other data types in the same list.

### 2. What is the meaning of `self` in Python?
**Answer**: `self` is a reference to the current instance of the class. It allows us to access variables and methods within the class. It's a way of referring to the instance within class methods.

### 3. What is the purpose of `__init__` in Python classes?
**Answer**: The `__init__` method is a special method in Python that is called when a new object of the class is created. It initializes the object's attributes.

### 4. How to make a class variable "private" using the `_` notation?
**Answer**: Using a single underscore (e.g., `_variable`) is a convention to indicate that a variable is intended to be private, though it is still accessible from outside the class. If you use double underscores (e.g., `__variable`), Python will name-mangle the variable, making it more challenging to access from outside.

### 5. What is `super()` in Python?
**Answer**: `super()` is used to call a method from a parent class within a child class. It allows you to call and extend the functionality of a method in the parent class.

### 6. If we want to print the method of the child class, why should we use inheritance in Python?
**Answer**: Inheritance is used when we want the child class to inherit properties and methods from the parent class, allowing code reuse and extension. If you want to override a method, you can do that in the child class using inheritance.

### 7. Can I define a class without inheritance but still use the same method as the parent?
**Answer**: Yes, you can define a class and have the same method, but without inheritance, the method will only exist in that specific class. Inheritance allows sharing and overriding methods across classes.

### 8. What is the purpose of the `in` operator in Python for dictionaries?
**Answer**: The `in` operator is used to check if a specific key exists in a dictionary.

### 9. How do we access the key-value pairs in a dictionary?
**Answer**: You can use a loop with the `items()` method to access all key-value pairs in a dictionary.

### 10. What is the meaning of Polymorphism in OOP?
**Answer**: Polymorphism allows objects of different classes to be treated as objects of a common base class. It also refers to the ability of a class to define methods that can be overridden in subclasses.

