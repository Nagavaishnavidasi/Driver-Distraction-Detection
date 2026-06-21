try:
    import tensorflow as tf
except ImportError:
    tf = None

keras = None
if tf is not None:
    keras = getattr(tf, "keras", None)

if keras is None:
    try:
        import keras
    except ImportError:
        keras = None

if tf is None:
    print("TensorFlow is not installed")
else:
    print("TensorFlow:", tf.__version__)

if keras is None:
    print("Keras is not installed")
else:
    print("Keras:", keras.__version__)