import math


class Value:
    def __init__(self, data, parents=(), op=""):
        self.data = float(data)
        self.grad = 0.0
        self.parents = parents
        self.op = op
        self.backward_rule = lambda: None

    def __repr__(self):
        return f"Value(data={self.data:.6f}, grad={self.grad:.6f}, op='{self.op}')"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def backward_rule():
            self.grad += out.grad
            other.grad += out.grad

        out.backward_rule = backward_rule
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def backward_rule():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out.backward_rule = backward_rule
        return out

    def __pow__(self, power):
        out = Value(self.data ** power, (self,), f"**{power}")

        def backward_rule():
            self.grad += power * self.data ** (power - 1) * out.grad

        out.backward_rule = backward_rule
        return out

    def relu(self):
        out = Value(max(0.0, self.data), (self,), "relu")

        def backward_rule():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad

        out.backward_rule = backward_rule
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + math.exp(-self.data))
        out = Value(s, (self,), "sigmoid")

        def backward_rule():
            self.grad += s * (1.0 - s) * out.grad

        out.backward_rule = backward_rule
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), "log")

        def backward_rule():
            self.grad += (1.0 / self.data) * out.grad

        out.backward_rule = backward_rule
        return out

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __truediv__(self, other):
        return self * other ** -1

    def backward(self):
        # Topological sort without recursion.
        order = []
        visited = set()
        stack = [(self, False)]

        while stack:
            node, parents_done = stack.pop()
            if parents_done:
                order.append(node)
                continue
            if node in visited:
                continue
            visited.add(node)
            stack.append((node, True))
            for parent in node.parents:
                if parent not in visited:
                    stack.append((parent, False))

        # Output gradient is 1; visit nodes in reverse order.
        self.grad = 1.0
        for node in reversed(order):
            node.backward_rule()


def value_sum(values):
    total = Value(0.0)
    for value in values:
        total = total + value
    return total


if __name__ == "__main__":
    a = Value(2.0)
    b = Value(-3.0)
    c = Value(10.0)

    y = ((a * b + c) ** 2).relu()
    y.backward()

    print("y:", y)
    print("a:", a)
    print("b:", b)
    print("c:", c)

    x = Value(3.0)
    z = x * x + x
    z.backward()
    print("Gradient of x*x + x at x=3 (expected 7):", x.grad)