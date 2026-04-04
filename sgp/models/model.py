from sgp.io import SGPIO

from .ann import ANN
from .mpd import MPD


class Model:
    """The prediction model"""

    #: the SGPIO object
    io: SGPIO
    #: the MPD model
    mpd: MPD
    #: the ANN model
    ann: ANN

    def __init__(self, io: SGPIO):
        """Initialize the model."""
        self.io = io
        self.mpd = MPD(io=io)
        self.ann = ANN(io=io)
