import keras.backend as K
from keras.layers import Layer


class Multiply(Layer):
    """Layer that takes element-wise product of a list of inputs.

    # Arguments
        **kwargs: standard layer keyword arguments.
    """

    def __init__(self, **kwargs):
        super(Multiply, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        if not isinstance(inputs, list):
            raise ValueError('A `Product` layer should be called on a list of inputs.')
        return inputs[0] * inputs[1]

    def compute_output_shape(self, input_shape):
        if not isinstance(input_shape, list):
            raise ValueError('A `Product` layer should be called on a list of inputs.')
        input_shapes = input_shape
        output_shape = list(input_shapes[0])
        return tuple(output_shape)


class Division(Layer):
    """Layer that takes element-wise product of a list of inputs.

    # Arguments
        **kwargs: standard layer keyword arguments.
    """

    def __init__(self, **kwargs):
        super(Division, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        if not isinstance(inputs, list):
            raise ValueError('A `Division` layer should be called on a list of inputs.')
        return (inputs[0] + K.epsilon()) / (inputs[1] + K.epsilon())

    def compute_output_shape(self, input_shape):
        if not isinstance(input_shape, list):
            raise ValueError('A `Division` layer should be called on a list of inputs.')
        input_shapes = input_shape
        output_shape = list(input_shapes[0])
        return tuple(output_shape)


class SignedDifference(Layer):
    """Layer that takes element-wise product of a list of inputs.

        # Arguments
            **kwargs: standard layer keyword arguments.
        """

    def __init__(self, **kwargs):
        super(SignedDifference, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        if not isinstance(inputs, list):
            raise ValueError('A `MetricA` layer should be called on a list of inputs.')
        return K.abs(inputs[0] - inputs[1]) / (inputs[0] * inputs[1] / K.abs(inputs[0] * inputs[1] + K.epsilon()))

    def compute_output_shape(self, input_shape):
        if not isinstance(input_shape, list):
            raise ValueError('A `MetricA` layer should be called on a list of inputs.')
        input_shapes = input_shape
        output_shape = list(input_shapes[0])
        return tuple(output_shape)


class Power(Layer):
    """Layer that takes the power of 2 of a input (dot-wise).

    # Arguments
        **kwargs: standard layer keyword arguments.
    """

    def __init__(self, order=2, **kwargs):
        super(Power, self).__init__(**kwargs)
        self.order = order

    def call(self, inputs, **kwargs):
        return K.pow(inputs, self.order)


class Diff(Layer):
    """
    # Arguments
       **kwargs: standard layer keyword arguments.
    """

    def __init__(self, **kwargs):
        super(Diff, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        if not isinstance(inputs, list):
            raise ValueError('A `Dif` layer should be called on a list of inputs.')
        return inputs[0] - inputs[1]

    def compute_output_shape(self, input_shape):
        if not isinstance(input_shape, list):
            raise ValueError('A `Dif` layer should be called on a list of inputs.')
        input_shapes = input_shape
        output_shape = list(input_shapes[0])
        return tuple(output_shape)


class Sum(Layer):
    """
    # Arguments
       **kwargs: standard layer keyword arguments.
    """

    def __init__(self, **kwargs):
        super(Sum, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        if not isinstance(inputs, list):
            raise ValueError('A `Sum` layer should be called on a list of inputs.')
        return inputs[0] + inputs[1]

    def compute_output_shape(self, input_shape):
        if not isinstance(input_shape, list):
            raise ValueError('A `Sum` layer should be called on a list of inputs.')
        input_shapes = input_shape
        output_shape = list(input_shapes[0])
        return tuple(output_shape)


class Abs(Layer):
    """Layer that takes the modulo (absolute value) of a input.

    # Arguments
        **kwargs: standard layer keyword arguments.
    """

    def __init__(self, **kwargs):
        super(Abs, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        return K.abs(inputs)
