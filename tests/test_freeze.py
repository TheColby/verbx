import numpy as np

from verbx.core.freeze import freeze_generator


def test_freeze_generator_exact_multiple() -> None:
    buf = np.array([[1], [2], [3], [4]])
    gen = freeze_generator(buf, block_size=2)

    block1 = next(gen)
    assert block1.shape == (2, 1)
    np.testing.assert_array_equal(block1, [[1], [2]])

    block2 = next(gen)
    assert block2.shape == (2, 1)
    np.testing.assert_array_equal(block2, [[3], [4]])

    block3 = next(gen)
    assert block3.shape == (2, 1)
    np.testing.assert_array_equal(block3, [[1], [2]])

def test_freeze_generator_wrap_around() -> None:
    buf = np.array([[1], [2], [3]])
    gen = freeze_generator(buf, block_size=2)

    block1 = next(gen)
    assert block1.shape == (2, 1)
    np.testing.assert_array_equal(block1, [[1], [2]])

    block2 = next(gen)
    assert block2.shape == (2, 1)
    np.testing.assert_array_equal(block2, [[3], [1]])

    block3 = next(gen)
    assert block3.shape == (2, 1)
    np.testing.assert_array_equal(block3, [[2], [3]])

    block4 = next(gen)
    assert block4.shape == (2, 1)
    np.testing.assert_array_equal(block4, [[1], [2]])

def test_freeze_generator_large_block() -> None:
    buf = np.array([[1], [2], [3]])
    gen = freeze_generator(buf, block_size=5)

    block1 = next(gen)
    assert block1.shape == (5, 1)
    np.testing.assert_array_equal(block1, [[1], [2], [3], [1], [2]])

    block2 = next(gen)
    assert block2.shape == (5, 1)
    np.testing.assert_array_equal(block2, [[3], [1], [2], [3], [1]])

def test_freeze_generator_empty_buffer_1d() -> None:
    buf = np.array([])
    gen = freeze_generator(buf, block_size=3)

    block1 = next(gen)
    assert block1.shape == (3,)
    np.testing.assert_array_equal(block1, [0.0, 0.0, 0.0])

    block2 = next(gen)
    assert block2.shape == (3,)
    np.testing.assert_array_equal(block2, [0.0, 0.0, 0.0])

def test_freeze_generator_empty_buffer_2d() -> None:
    buf = np.zeros((0, 2))
    gen = freeze_generator(buf, block_size=3)

    block1 = next(gen)
    assert block1.shape == (3, 2)
    np.testing.assert_array_equal(block1, [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])

def test_freeze_generator_1d_input() -> None:
    buf = np.array([1, 2, 3])
    gen = freeze_generator(buf, block_size=2)

    block1 = next(gen)
    assert block1.shape == (2,)
    np.testing.assert_array_equal(block1, [1, 2])

    block2 = next(gen)
    assert block2.shape == (2,)
    np.testing.assert_array_equal(block2, [3, 1])

def test_freeze_generator_2d_stereo() -> None:
    buf = np.array([[1, 10], [2, 20], [3, 30]])
    gen = freeze_generator(buf, block_size=4)

    block1 = next(gen)
    assert block1.shape == (4, 2)
    np.testing.assert_array_equal(block1, [[1, 10], [2, 20], [3, 30], [1, 10]])

    block2 = next(gen)
    assert block2.shape == (4, 2)
    np.testing.assert_array_equal(block2, [[2, 20], [3, 30], [1, 10], [2, 20]])
