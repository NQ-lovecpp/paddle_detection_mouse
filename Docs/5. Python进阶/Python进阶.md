**内置函数与特殊方法**：
   补充一些内置函数和特殊方法，例如`__str__`, `__repr__`, `__len__`等的使用。
# 一、Python中一切皆是对象

对象是指在内存中具有唯一标识符、类型和值的实例。换句话说，它是一个具有属性和方法的实体，这些属性可以被访问，方法可以被调用。

```python
num = 10
# 打印num的唯一标识符，其实是内存地址
print(id(num))
```

## 函数和类也是对象，属于Python的一等公民

>函数是对象，因此可以打印出某个函数的唯一标识符

```python
def print_value(name):
    print(name)

print(id(print_value))

my_func = print_value
```

>类本身是对象，因此可以打印出类的唯一标识符，每次打印都是一样的，而类的对象也是对象，但每个不同临时对象打印出的唯一标识符不一样

```python
class Person:     
	def __init__(self):         
		print("Person()") 

print(id(Person())) 
print(id(Person())) 
print(id(Person())) 
print(id(Person())) 

print("-----") 

print(id(Person)) 
print(id(Person)) 
print(id(Person)) 
print(id(Person))
```

![[Pasted image 20240703094921.png]]

## 对象的属性和方法

在Python中，类可以有属性和方法。属性是对象的状态，而方法是对象的行为。

```python
class Person:
	def __init__(self, name, age):
		self.name = name
		self.age = age

	def greet(self):
		return f"Hello, my name is {self.name} and I am {self.age} years old."

p = Person("Alice", 30)
print(p.name)  # 访问属性
print(p.greet())  # 调用方法
```

## 类和实例的区别

类是对象的蓝图，而实例是通过类创建的具体对象。

```python
class Dog:
    def __init__(self, name):
        self.name = name

d1 = Dog("Rex")
d2 = Dog("Buddy")

print(id(Dog))  # 打印类的唯一标识符
print(id(d1))   # 打印实例对象的唯一标识符
print(id(d2))   # 打印实例对象的唯一标识符
```

## 对象的生命周期

对象的创建和销毁由构造函数`__init__`和析构函数`__del__`负责。

```python
class Car:
    def __init__(self, model):
        self.model = model
        print(f"{self.model} is created.")

    def __del__(self):
        print(f"{self.model} is destroyed.")

c = Car("Toyota")
del c  # 手动销毁对象
```

## 内置函数与特殊方法

Python有许多内置函数和特殊方法，用于操作对象。

```python
class Example:
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return f"Example with value: {self.value}"

    def __len__(self):
        return len(self.value)

e = Example([1, 2, 3])
print(str(e))  # 调用__str__方法
print(len(e))  # 调用__len__方法
```


##  type object class的关系

```python
class Student:
    pass

# 探究对象是被谁创建的
stu = Student()
print(stu)
print(type(stu))
print(type(Student))
```

```python
class MyStudent:
    pass

# object类是所有类的父类
print(int.__base__)
print(str.__base__)

# 自定义类型，如果没有继承体系，object类是它的父类
print(MyStudent.__base__)
```

>[!Attention] 注意：
>Python的元类和类型系统：
>1. object是被type创建的
>2. type继承了object
>3. type创建了所有的对象，也包含类对象（object）
>4. type在创建object基类同时也继承了object
>5. type是由自身创建的
> ---
>### 1. `object`是被`type`创建的
>
>在Python中，所有的类都是由`type`这个元类创建的，`object`也是一个类，它也是由`type`创建的。
>
>```python
># `object`类的类型是`type`
>print(type(object))  # 输出：<class 'type'>
>```
>
>### 2. `type`继承了`object`
>
>`type`是一个类，它也继承了`object`，这意味着`type`本身也是一个对象。
>
>```python
># `type`类继承自`object`
>print(issubclass(type, object))  # 输出：True
>```
>
>### 3. `type`创建了所有的对象，也包含类对象（object）
>
>在Python中，所有的类（包括内置类和用户定义的类）都是由`type`创建的。
>
>```python
>class MyClass:
>    pass
>
># `MyClass`的类型是`type`
>print(type(MyClass))  # 输出：<class 'type'>
>```
>
>### 4. `type`在创建`object`基类同时也继承了`object`
>
>`type`是一个类，它在创建`object`类时，也继承了`object`类。
>
>```python
># `type`的类型是`type`，并且它继承自`object`
>print(type(type))  # 输出：<class 'type'>
>print(issubclass(type, object))  # 输出：True
>```
>
>### 5. `type`是由自身创建的
>
>`type`是一个特殊的元类，它是由自身创建的，这形成了一个自引用。
>
>```python
># `type`的类型是`type`
>print(type(type))  # 输出：<class 'type'>
>```





# 二、魔术方法

## 1. 什么是魔术方法

魔术方法（Magic Methods），也称为特殊方法（Special Methods）或双下划线方法（Dunder Methods），是由双下划线（`__`）包围的方法。它们使得类的实例能够与Python内置操作进行交互，比如迭代、切片、上下文管理、数学运算等。魔术方法通常由Python解释器隐式调用，不需要在代码中显式调用。

>被双下划线修饰的方法被称为“魔术方法”

- `__init__(self, ...)`：构造函数，在创建对象时调用。
- `__str__(self)`：定义对象的字符串表示，在使用`print`函数或`str()`时调用。
- `__repr__(self)`：定义对象的正式字符串表示，通常在调试和交互式解释器中使用。
- `__len__(self)`：在使用`len()`函数时调用。
- `__getitem__(self, key)`：允许对象使用下标访问（即使对象像列表一样可索引）。

## 2. `__getitem__` 魔术方法

`__getitem__` 魔术方法使得对象可以像列表、字典等容器类型一样，通过下标进行访问。它允许我们定义如何获取特定键或索引对应的值。

```python
# 被双下划线修饰的方法被称为“魔术方法”

class Student:
    def __init__(self, student_list) -> None:
        self._student_list = student_list

    # 让这个类拥有序列特征，可以通过下标访问元素
    def __getitem__(self, item):
        return self._student_list[item]


student = Student(['chen','zhuo','haha','你好',123])

# 迭代student中的列表
for stu in student._student_list:
    print(stu)

print("------")

# 直接迭代student对象
for stu in student:
    print(stu)

print("------")

# 下标访问
print(student[0], student[1], student[3])
```

![[Pasted image 20240703112636.png]]


在这个示例中，我们创建了一个 `Student` 类，并在该类中实现了 `__getitem__` 魔术方法。

- `__init__(self, student_list)`：构造函数，接受一个学生名单并将其存储在实例变量 `_student_list` 中。
- `__getitem__(self, item)`：实现下标访问，使得 `Student` 实例可以通过下标访问内部的 `_student_list`。

>[!Tip] 通过实现 `__getitem__` 方法，我们可以：
>
>1. 直接迭代 `Student` 实例，就像迭代一个列表一样。
>2. 使用下标访问 `Student` 实例中的元素，就像访问列表中的元素一样。

解释一下原理：

通过实现`__getitem__`方法，使得自定义类的实例可以使用迭代特性。当实现了`__getitem__`方法并且类实例被迭代时，Python会从索引0开始调用`__getitem__`方法，依次获取元素，直到引发`IndexError`异常为止。



# 三、深入类和对象
# 第四章自定义序列类
# 第五章深入python的set和dict
# 第六章对象引用、可变性和垃圾回收