# Programming Reference Book

## Table of Contents
1. [Programming Fundamentals](#programming-fundamentals)
2. [Data Structures](#data-structures)
3. [Algorithms](#algorithms)
4. [Git Commands](#git-commands)
5. [Command Line Reference](#command-line-reference)
6. [Web Development](#web-development)
7. [Database Operations](#database-operations)
8. [Testing](#testing)
9. [Security Best Practices](#security-best-practices)
10. [Common Patterns](#common-patterns)

---

## Programming Fundamentals

### Variables and Data Types

#### JavaScript
```javascript
// Variables
let name = "John";
const age = 25;
var isActive = true;

// Data Types
let number = 42;
let string = "Hello World";
let boolean = true;
let array = [1, 2, 3];
let object = { key: "value" };
let nullValue = null;
let undefinedValue = undefined;
```

#### Python
```python
# Variables
name = "John"
age = 25
is_active = True

# Data Types
number = 42
string = "Hello World"
boolean = True
list_data = [1, 2, 3]
dict_data = {"key": "value"}
none_value = None
```

### Control Structures

#### Conditionals
```javascript
// JavaScript
if (condition) {
    // code
} else if (anotherCondition) {
    // code
} else {
    // code
}

// Ternary operator
let result = condition ? "true value" : "false value";
```

```python
# Python
if condition:
    # code
elif another_condition:
    # code
else:
    # code

# Ternary operator
result = "true value" if condition else "false value"
```

#### Loops
```javascript
// JavaScript
for (let i = 0; i < 10; i++) {
    console.log(i);
}

// For...of loop
for (const item of array) {
    console.log(item);
}

// While loop
while (condition) {
    // code
}
```

```python
# Python
for i in range(10):
    print(i)

# For loop with list
for item in list_data:
    print(item)

# While loop
while condition:
    # code
```

### Functions

#### JavaScript
```javascript
// Function declaration
function greet(name) {
    return `Hello, ${name}!`;
}

// Arrow function
const greet = (name) => `Hello, ${name}!`;

// Async function
async function fetchData() {
    const response = await fetch('/api/data');
    return response.json();
}
```

#### Python
```python
# Function definition
def greet(name):
    return f"Hello, {name}!"

# Lambda function
greet = lambda name: f"Hello, {name}!"

# Async function
async def fetch_data():
    response = await client.get('/api/data')
    return response.json()
```

---

## Data Structures

### Arrays/Lists

#### JavaScript Arrays
```javascript
// Creation
let arr = [1, 2, 3, 4, 5];

// Common methods
arr.push(6);           // Add to end
arr.pop();             // Remove from end
arr.unshift(0);        // Add to beginning
arr.shift();           // Remove from beginning
arr.splice(2, 1);      // Remove at index
arr.slice(1, 3);       // Extract portion
arr.indexOf(3);        // Find index
arr.includes(4);       // Check existence

// Iteration methods
arr.forEach(item => console.log(item));
arr.map(item => item * 2);
arr.filter(item => item > 2);
arr.reduce((sum, item) => sum + item, 0);
```

#### Python Lists
```python
# Creation
arr = [1, 2, 3, 4, 5]

# Common methods
arr.append(6)          # Add to end
arr.pop()              # Remove from end
arr.insert(0, 0)       # Add at index
arr.remove(3)          # Remove first occurrence
arr.index(4)           # Find index
4 in arr               # Check existence

# List comprehension
doubled = [item * 2 for item in arr]
filtered = [item for item in arr if item > 2]
```

### Objects/Dictionaries

#### JavaScript Objects
```javascript
// Creation
let obj = {
    name: "John",
    age: 25,
    greet() {
        return `Hello, I'm ${this.name}`;
    }
};

// Access
obj.name;              // Dot notation
obj["age"];            // Bracket notation

// Manipulation
obj.city = "New York"; // Add property
delete obj.age;        // Delete property

// Object methods
Object.keys(obj);      // Get keys
Object.values(obj);    // Get values
Object.entries(obj);   // Get key-value pairs
```

#### Python Dictionaries
```python
# Creation
obj = {
    "name": "John",
    "age": 25
}

# Access
obj["name"]            # Bracket notation
obj.get("age", 0)      # Safe access with default

# Manipulation
obj["city"] = "New York"  # Add key-value
del obj["age"]         # Delete key

# Dictionary methods
obj.keys()             # Get keys
obj.values()           # Get values
obj.items()            # Get key-value pairs
```

---

## Algorithms

### Sorting Algorithms

#### Quick Sort
```python
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)
```

#### Merge Sort
```python
def mergesort(arr):
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = mergesort(arr[:mid])
    right = mergesort(arr[mid:])
    
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result
```

### Search Algorithms

#### Binary Search
```python
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1
```

---

## Git Commands

### Basic Commands
```bash
# Repository setup
git init                    # Initialize repository
git clone <url>             # Clone repository
git remote add origin <url> # Add remote

# Status and changes
git status                  # Check status
git diff                    # Show changes
git diff --staged           # Show staged changes

# Staging and committing
git add <file>              # Stage file
git add .                   # Stage all files
git commit -m "message"     # Commit with message
git commit -am "message"    # Stage and commit

# Branching
git branch                  # List branches
git branch <name>           # Create branch
git checkout <branch>       # Switch branch
git checkout -b <branch>    # Create and switch
git merge <branch>          # Merge branch
git branch -d <branch>      # Delete branch

# Remote operations
git push                    # Push to remote
git pull                    # Pull from remote
git fetch                   # Fetch changes

# History
git log                     # View commit history
git log --oneline           # Compact log
git show <commit>           # Show commit details
```

---

## Command Line Reference

### Navigation
```bash
pwd                        # Print working directory
ls                         # List files
ls -la                     # List with details
cd <directory>             # Change directory
cd ..                      # Go up one level
cd ~                       # Go to home directory
```

### File Operations
```bash
mkdir <directory>          # Create directory
rmdir <directory>          # Remove empty directory
rm <file>                  # Remove file
rm -rf <directory>         # Remove directory recursively
cp <source> <dest>         # Copy file
mv <source> <dest>         # Move/rename file
touch <file>               # Create empty file
```

### File Content
```bash
cat <file>                 # Display file content
less <file>                # View file with pagination
head <file>                # Show first 10 lines
tail <file>                # Show last 10 lines
tail -f <file>             # Follow file changes
grep <pattern> <file>      # Search in file
find <path> -name <pattern> # Find files
```

### Process Management
```bash
ps                         # List running processes
ps aux                     # Detailed process list
top                        # Real-time process monitor
htop                       # Enhanced process monitor
kill <pid>                 # Kill process by ID
killall <name>             # Kill processes by name
```

---

## Web Development

### HTML Structure
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Page Title</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <header>
        <nav>
            <ul>
                <li><a href="#home">Home</a></li>
                <li><a href="#about">About</a></li>
            </ul>
        </nav>
    </header>
    
    <main>
        <section>
            <h1>Main Heading</h1>
            <p>Content paragraph</p>
        </section>
    </main>
    
    <footer>
        <p>&copy; 2024 Website Name</p>
    </footer>
    
    <script src="script.js"></script>
</body>
</html>
```

### CSS Basics
```css
/* Selectors */
.class-name { }
#id-name { }
element { }
element.class { }
element > child { }
element + sibling { }

/* Common Properties */
.container {
    display: flex;
    justify-content: center;
    align-items: center;
    width: 100%;
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
    background-color: #f5f5f5;
    border: 1px solid #ddd;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

/* Responsive Design */
@media (max-width: 768px) {
    .container {
        flex-direction: column;
        padding: 10px;
    }
}
```

### JavaScript DOM Manipulation
```javascript
// Selecting elements
const element = document.getElementById('id');
const elements = document.querySelectorAll('.class');
const element = document.querySelector('.class');

// Creating elements
const newDiv = document.createElement('div');
newDiv.textContent = 'Hello World';
newDiv.classList.add('my-class');

// Adding to DOM
document.body.appendChild(newDiv);

// Event handling
element.addEventListener('click', function(e) {
    e.preventDefault();
    console.log('Element clicked');
});

// Fetch API
async function fetchData() {
    try {
        const response = await fetch('/api/data');
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Error:', error);
    }
}
```

---

## Database Operations

### SQL Basics
```sql
-- Create table
CREATE TABLE users (
    id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(100) UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert data
INSERT INTO users (name, email) VALUES ('John Doe', 'john@email.com');

-- Select data
SELECT * FROM users;
SELECT name, email FROM users WHERE age > 18;
SELECT COUNT(*) FROM users;

-- Update data
UPDATE users SET name = 'Jane Doe' WHERE id = 1;

-- Delete data
DELETE FROM users WHERE id = 1;

-- Joins
SELECT u.name, p.title 
FROM users u 
JOIN posts p ON u.id = p.user_id;
```

### MongoDB Operations
```javascript
// Connect to database
const { MongoClient } = require('mongodb');
const client = new MongoClient(url);

// Insert document
await collection.insertOne({
    name: 'John Doe',
    email: 'john@email.com'
});

// Find documents
const users = await collection.find({ age: { $gt: 18 } }).toArray();

// Update document
await collection.updateOne(
    { _id: userId },
    { $set: { name: 'Jane Doe' } }
);

// Delete document
await collection.deleteOne({ _id: userId });
```

---

## Testing

### Unit Testing with Jest
```javascript
// test.js
const sum = require('./sum');

describe('Sum function', () => {
    test('adds 1 + 2 to equal 3', () => {
        expect(sum(1, 2)).toBe(3);
    });
    
    test('handles negative numbers', () => {
        expect(sum(-1, 1)).toBe(0);
    });
    
    test('throws error for invalid input', () => {
        expect(() => {
            sum('a', 'b');
        }).toThrow();
    });
});

// Async testing
test('async function test', async () => {
    const data = await fetchData();
    expect(data).toBeDefined();
});

// Mock testing
jest.mock('./api');
const mockFetch = require('./api');
mockFetch.getData.mockResolvedValue({ id: 1, name: 'Test' });
```

### Python Testing with pytest
```python
import pytest

def test_sum():
    assert sum([1, 2, 3]) == 6

def test_division_by_zero():
    with pytest.raises(ZeroDivisionError):
        result = 1 / 0

@pytest.fixture
def sample_data():
    return {"name": "Test", "value": 42}

def test_with_fixture(sample_data):
    assert sample_data["name"] == "Test"
    assert sample_data["value"] == 42

@pytest.mark.asyncio
async def test_async_function():
    result = await async_function()
    assert result is not None
```

---

## Security Best Practices

### Input Validation
```javascript
// Sanitize user input
const validator = require('validator');

function validateEmail(email) {
    return validator.isEmail(email);
}

function sanitizeHtml(input) {
    return validator.escape(input);
}

// Prevent SQL injection
const query = 'SELECT * FROM users WHERE id = ?';
db.query(query, [userId], (err, results) => {
    // Safe parameterized query
});
```

### Authentication & Authorization
```javascript
const bcrypt = require('bcrypt');
const jwt = require('jsonwebtoken');

// Hash password
async function hashPassword(password) {
    const saltRounds = 12;
    return await bcrypt.hash(password, saltRounds);
}

// Verify password
async function verifyPassword(password, hash) {
    return await bcrypt.compare(password, hash);
}

// JWT token
function generateToken(userId) {
    return jwt.sign({ userId }, process.env.JWT_SECRET, { expiresIn: '1h' });
}

// Verify token middleware
function authenticateToken(req, res, next) {
    const token = req.header('Authorization')?.replace('Bearer ', '');
    
    if (!token) {
        return res.status(401).json({ error: 'Access denied' });
    }
    
    try {
        const decoded = jwt.verify(token, process.env.JWT_SECRET);
        req.userId = decoded.userId;
        next();
    } catch (error) {
        res.status(403).json({ error: 'Invalid token' });
    }
}
```

### Environment Variables
```bash
# .env file
DATABASE_URL=postgresql://user:pass@localhost:5432/db
JWT_SECRET=your-secret-key
API_KEY=your-api-key
NODE_ENV=development
```

```javascript
// Using environment variables
require('dotenv').config();

const config = {
    port: process.env.PORT || 3000,
    database: process.env.DATABASE_URL,
    jwtSecret: process.env.JWT_SECRET
};
```

---

## Common Patterns

### Design Patterns

#### Singleton Pattern
```javascript
class Singleton {
    constructor() {
        if (Singleton.instance) {
            return Singleton.instance;
        }
        Singleton.instance = this;
    }
    
    static getInstance() {
        if (!Singleton.instance) {
            Singleton.instance = new Singleton();
        }
        return Singleton.instance;
    }
}
```

#### Observer Pattern
```javascript
class EventEmitter {
    constructor() {
        this.events = {};
    }
    
    on(event, listener) {
        if (!this.events[event]) {
            this.events[event] = [];
        }
        this.events[event].push(listener);
    }
    
    emit(event, data) {
        if (this.events[event]) {
            this.events[event].forEach(listener => listener(data));
        }
    }
    
    off(event, listenerToRemove) {
        if (this.events[event]) {
            this.events[event] = this.events[event].filter(
                listener => listener !== listenerToRemove
            );
        }
    }
}
```

#### Factory Pattern
```javascript
class UserFactory {
    static createUser(type, data) {
        switch (type) {
            case 'admin':
                return new AdminUser(data);
            case 'regular':
                return new RegularUser(data);
            default:
                throw new Error('Invalid user type');
        }
    }
}
```

### Error Handling

#### JavaScript
```javascript
// Try-catch for sync code
try {
    const result = riskyOperation();
    console.log(result);
} catch (error) {
    console.error('Error occurred:', error.message);
} finally {
    // Cleanup code
    console.log('Cleanup completed');
}

// Promise error handling
fetchData()
    .then(data => processData(data))
    .catch(error => console.error('Error:', error))
    .finally(() => console.log('Done'));

// Async/await error handling
async function handleAsyncOperation() {
    try {
        const data = await fetchData();
        const processed = await processData(data);
        return processed;
    } catch (error) {
        console.error('Async error:', error);
        throw error; // Re-throw if needed
    }
}
```

#### Python
```python
# Try-except for error handling
try:
    result = risky_operation()
    print(result)
except ValueError as e:
    print(f"Value error: {e}")
except Exception as e:
    print(f"General error: {e}")
else:
    print("No errors occurred")
finally:
    print("Cleanup completed")

# Context manager
with open('file.txt', 'r') as f:
    content = f.read()
# File automatically closed
```

### Regular Expressions

#### Common Patterns
```javascript
// Email validation
const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

// Phone number (US format)
const phoneRegex = /^\(\d{3}\)\s\d{3}-\d{4}$/;

// URL validation
const urlRegex = /^https?:\/\/[^\s/$.?#].[^\s]*$/;

// Password strength (8+ chars, 1 upper, 1 lower, 1 digit)
const passwordRegex = /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)[a-zA-Z\d@$!%*?&]{8,}$/;

// Usage
function validateInput(input, regex) {
    return regex.test(input);
}
```

---

## Performance Optimization

### JavaScript Optimization
```javascript
// Debouncing
function debounce(func, delay) {
    let timeoutId;
    return function (...args) {
        clearTimeout(timeoutId);
        timeoutId = setTimeout(() => func.apply(this, args), delay);
    };
}

// Throttling
function throttle(func, delay) {
    let lastCall = 0;
    return function (...args) {
        const now = Date.now();
        if (now - lastCall >= delay) {
            lastCall = now;
            func.apply(this, args);
        }
    };
}

// Memoization
function memoize(fn) {
    const cache = new Map();
    return function (...args) {
        const key = JSON.stringify(args);
        if (cache.has(key)) {
            return cache.get(key);
        }
        const result = fn.apply(this, args);
        cache.set(key, result);
        return result;
    };
}
```

---

## Useful Resources

### Documentation Links
- [MDN Web Docs](https://developer.mozilla.org/)
- [Python Documentation](https://docs.python.org/)
- [Git Documentation](https://git-scm.com/docs)
- [React Documentation](https://reactjs.org/docs)
- [Node.js Documentation](https://nodejs.org/docs)

### Package Managers
```bash
# npm (Node.js)
npm init                   # Initialize package.json
npm install <package>      # Install package
npm install -g <package>   # Install globally
npm run <script>           # Run script from package.json

# pip (Python)
pip install <package>      # Install package
pip install -r requirements.txt  # Install from requirements
pip freeze > requirements.txt    # Save dependencies

# yarn (Alternative to npm)
yarn init                  # Initialize package.json
yarn add <package>         # Install package
yarn run <script>          # Run script
```

---

*This reference book covers essential programming concepts, commands, and patterns. Keep it handy for quick reference during development!*

**Last updated:** August 2024