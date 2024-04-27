# %% matplotlib inline
from jaxtyping import Float, jaxtyped
from typeguard import typechecked as typechecker
import numpy as np
import matplotlib.pyplot as plt


class ReLU:
    @jaxtyped(typechecker=typechecker)
    def __call__(self, x: Float[np.ndarray, "*dims"]) -> Float[np.ndarray, "*dims"]:
        x[x < 0] = 0
        return x


class Linear:
    def __init__(self, in_dim: int, out_dim: int) -> None:
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.b = np.zeros([1, out_dim])
        # this is def wrong
        he_bound = 2 / in_dim
        self.w = np.random.rand(in_dim, out_dim) * 2 * he_bound - he_bound
        self.f = None  # pre-activations
        self.dw = None
        self.db = None

    def set_grads(self, df, inputs):
        db = np.sum(df, axis=0)
        dw = inputs.T @ df

        assert db.shape == (self.b.shape[-1],)
        assert dw.shape == self.w.shape

        self.db = db
        self.dw = dw

    def step(self, lr: float):
        assert self.dw is not None and self.db is not None
        self.w = self.w - lr * self.dw
        self.b = self.b - lr * self.db

        self.dw = None
        self.db = None

    @jaxtyped(typechecker=typechecker)
    def __call__(
        self, X: Float[np.ndarray, "*batch {self.in_dim}"]
    ) -> Float[np.ndarray, "*batch {self.out_dim}"]:
        out = X @ self.w + self.b
        self.f = out
        return out


def relu_derivative(x: np.ndarray):
    new_x = x.copy()
    new_x[new_x < 0] = 0
    new_x[new_x >= 0] = 1
    return new_x


class MLP:
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: int = 1,
        learning_rate: float = 0.005,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.relu = ReLU()
        self.hidden_layers = [Linear(input_dim, hidden_dim)]
        if n_layers > 1:
            self.hidden_layers += [
                Linear(hidden_dim, hidden_dim) for _ in range(n_layers - 1)
            ]
        self.output_layer = Linear(hidden_dim, output_dim)
        self.inputs = None
        self.lr = learning_rate

    def all_layers(self) -> list[Linear]:
        return [*self.hidden_layers, self.output_layer]

    @jaxtyped(typechecker=typechecker)
    def __call__(
        self, X: Float[np.ndarray, "*batch {self.input_dim}"]
    ) -> Float[np.ndarray, "*batch {self.output_dim}"]:
        self.inputs = X
        for layer in self.hidden_layers:
            X = self.relu(layer(X))
        X = self.output_layer(X)
        return X

    @jaxtyped(typechecker=typechecker)
    def backwards(self, Y: Float[np.ndarray, "*batch {self.output_dim}"]):
        layer_prev: Linear | None = None
        df_prev: np.ndarray | None = None

        layers = list(reversed(self.all_layers()))
        for i, layer in enumerate(layers):
            if i == 0:
                df = (2 / Y.shape[0]) * (layer.f - Y)
            else:
                assert layer_prev is not None and df_prev is not None
                df = relu_derivative(layer.f) * (df_prev @ layer_prev.w.T)

            if i == len(layers) - 1:
                layer.set_grads(df, self.inputs)
            else:
                layer.set_grads(df, self.relu(layers[i + 1].f))

            layer_prev = layer
            df_prev = df

    def step(self):
        for layer in self.all_layers():
            layer.step(self.lr)


def least_squares(preds: np.ndarray, labels: np.ndarray) -> float:
    squares = (preds - labels) ** 2
    loss = np.mean(squares)
    return loss


def train(network, X, Y, loss_fn=least_squares, batch_size=1, epochs=20000):
    # Y = np.expand_dims(Y, 1)
    Y = Y.astype(np.float64)[:, None]
    for i in range(epochs):
        copy = X.copy()
        preds = network(copy)
        loss = loss_fn(preds, Y)
        if i % 100 == 0:
            print(f"{i} -> {loss}")
        network.backwards(Y)
        network.step()


def xor(x):
    a, b = x
    return (a and not b) or (not a and b)


def plot_xor(X, preds, jitter=0.03, s=2):
    # Flatten the preds array to one dimension
    preds = preds.flatten()

    # Add jitter: a small random noise to make the points more distinguishable
    jitter_x = np.random.uniform(-jitter, jitter, X.shape[0])
    jitter_y = np.random.uniform(-jitter, jitter, X.shape[0])

    # Now preds can be used for boolean indexing
    class_0 = X[preds >= 0.5]
    class_1 = X[preds < 0.5]

    # Plot the points with jitter
    plt.scatter(
        class_0[:, 0] + jitter_x[preds >= 0.5],
        class_0[:, 1] + jitter_y[preds >= 0.5],
        color="red",
        label="Class 0",
        s=s,
    )
    plt.scatter(
        class_1[:, 0] + jitter_x[preds < 0.5],
        class_1[:, 1] + jitter_y[preds < 0.5],
        color="blue",
        label="Class 1",
        s=s,
    )
    plt.xlabel("Input 1")
    plt.ylabel("Input 2")
    plt.title("XOR Dataset with Jitter")
    plt.legend()
    plt.show()


# X = np.linspace(0, 10, 100)
# X = np.expand_dims(X, axis=1)
# Y = np.sin(X)
X = np.random.rand(1000, 2)
X[X < 0.5] = 0
X[X >= 0.5] = 1
Y = np.apply_along_axis(xor, 1, X)
plot_xor(X, Y)

shallow = MLP(2, 5, 1)
untrained_preds = shallow(X.copy())
plot_xor(X, untrained_preds)
train(shallow, X, Y)
trained_preds = shallow(X.copy())
plot_xor(X, trained_preds)

# print(shallow)
# input = np.array([1, 2])
# print(input)
# print(preds)
# print(Y[1])
# print(out)
