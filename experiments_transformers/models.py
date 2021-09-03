from transformerModels.transformerDecoder import TransformerDecoderModel
from transformerModels.transformerDecoder2 import TransformerDecoderModel2
from transformerModels.transformerEncoderModel import TransformerEncoderModel


def TransformerDecoder(
    input_shape,
    output_size,
    N=3,
    d_model=256,
    h=8):

    model = TransformerDecoderModel(input_shape[-2],output_size,input_shape[-1],d_model,h,N)

    return model

def TransformerDecoder2(
    input_shape,
    output_size,
    N=3,
    d_model=256,
    h=8):

    model = TransformerDecoderModel2(input_shape[-2],output_size,input_shape[-1],d_model,h,N)

    return model

def TransformerEncoder(
    input_shape,
    output_size,
    N=3,
    d_model=256,
    h=8):

    model = TransformerEncoderModel(input_shape[-2],output_size,input_shape[-1],d_model,h,N)

    return model

model_factory = {
    "trD_AR": TransformerDecoder,
    "trD2_AR": TransformerDecoder2,
    "trE": TransformerEncoder
}


def create_model(model_name, input_shape, **args):
    assert model_name in model_factory.keys(), "Model '{}' not supported".format(
        model_name
    )
    return model_factory[model_name](input_shape, **args)
