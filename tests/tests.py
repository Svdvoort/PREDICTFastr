import PREDICT
import os

def test_calcfeatures():
    # Configure location of input
    current_path = os.path.dirname(os.path.abspath(__file__))
    image = os.path.join(current_path,
                        'ExampleData', 'ExampleImage.nii.gz')
    segmentation = os.path.join(current_path,
                                'ExampleData', 'ExampleMask.nii.gz')
    metadata = os.path.join(current_path,
                            'ExampleData', 'ExampleDCM.dcm')
    config_predict = os.path.join(current_path,
                                'ExampleData', 'config.ini')


    # Configure location of output
    output_predict = os.path.join(current_path,
                                'ExampleData',
                                'ExampleFeaturesPREDICT.hdf5')

    # Extract PREDICT features
    PREDICT.CalcFeatures.CalcFeatures(image=image, segmentation=segmentation,
                                    parameters=config_predict,
                                    metadata_file=metadata,
                                    output=output_predict)
    
    
if __name__ == "__main__":
    test_calcfeatures()