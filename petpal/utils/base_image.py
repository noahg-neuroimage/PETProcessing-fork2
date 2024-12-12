"""
Class for managing image data.
"""
import json
import numpy as np
import nibabel


class BaseImage(np.lib.mixins.NDArrayOperatorsMixin):
    """
    Base image class for handling images in PETPAL.
    """
    def __init__(self, data):

        self.data = np.asarray(data)

    def __repr__(self):

        return f"{self.__class__.__name__}(data={self.data})"

    def __array__(self, dtype=None, copy=None):

        if copy is False:

            raise ValueError(

                "`copy=False` isn't supported. A copy is always created."

            )

        return self.data

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):

        if method == '__call__':

            return self.__class__(ufunc(*inputs, **kwargs))

        else:

            return NotImplemented

    def __len__(self):

        return len(self.data)

    def __getitem__(self,a):

        return self.data[a]

    @property
    def shape(self):
        """
        Shape of the data array
        """
        return self.data.shape



class HeaderImage(BaseImage):
    """
    Image class to handle image data and header
    """
    def __init__(self,data,header: dict):
        BaseImage.__init__(self,data)
        self.header = header

    def __repr__(self):

        return f"{self.__class__.__name__}(data={self.data},header={self.header})"

class BidsMeta:
    """
    Base class to handle bids metadata
    """
    def __init__(self):
        pass


def load(image_path: str):
    """
    Load image data and header
    """
    image = nibabel.load(image_path)
    header_image = HeaderImage(data=image.get_fdata(),header=image.header)
    return header_image


def load_bidsmeta(meta_path) -> BidsMeta:
    """
    Load bids metadata
    """
    with open(meta_path, 'r', encoding='utf-8') as meta_file:
        image_meta = json.load(meta_file)
    bidsmeta = BidsMeta()
    for key in image_meta.keys():
        # TODO: find a better way to do this
        bidsmeta.__setattr__(key.lower(),image_meta[key])
    return bidsmeta


class BidsImage(HeaderImage,BidsMeta):
    """
    Image class that implements image data, header, and bids metadata
    """
    def __init__(self,data,header: dict,bidsmeta: BidsMeta):
        HeaderImage.__init__(self,data,header)
        BidsMeta.__init__(self)
        self.bidsmeta = bidsmeta
    
    def __repr__(self):

        return f"{self.__class__.__name__}(data={self.data},header={self.header},bidsmeta={self.bidsmeta})"
    
    def image_decorator(self,func,*args,**kwargs):
        data = self.data
        result = func(data,*args,**kwargs)


class PetImage4d(BidsImage):
    """
    Image class that implements BidsImage as well as methods specific to 4D PET data, such as half
    life.
    """
    def __init__(self,data,header: dict,bidsmeta: BidsMeta):
        BidsImage.__init__(self,data,header,bidsmeta)

    def __repr__(self):
        return f"{self.__class__.__name__}(data={self.data},header={self.header},bidsmeta={self.bidsmeta})"

    @property
    def half_life(self):
        """
        Radioisotope half life in seconds
        """
        return 6.58404*1000 # TODO: figure out best way to store values and convert from bidsmeta.TracerRadionuclide

    def validate_required_fields(self):
        # idea: useful but might be better elsewhere
        # go through fields required for PET and validate at least those used by petpal software
        pass
