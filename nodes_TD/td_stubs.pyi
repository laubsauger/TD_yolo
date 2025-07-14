"""
TouchDesigner Python API type stubs
Based on TouchDesigner 2025.30000+
"""
from typing import Any, List, Optional, Union, Tuple

# Core TouchDesigner types
class OP:
    """Base operator class"""
    name: str
    path: str
    id: int
    digits: int
    par: 'ParCollection'
    parent: Optional['OP']
    time: 'TimeCOMP'
    
    def __getitem__(self, key: Union[str, int]) -> 'Channel':
        """Access channel by name or index"""
        ...
    
    def cook(self, force: bool = False) -> None:
        """Force cook of operator"""
        ...
    
    def eval(self) -> Any:
        """Evaluate operator"""
        ...

class TOP(OP):
    """Texture operator"""
    width: int
    height: int
    
    def numpyArray(self, delayed: bool = True, writable: bool = False) -> Any:
        """Get numpy array from TOP"""
        ...
    
    def copyNumpyArray(self, array: Any) -> None:
        """Copy numpy array to TOP"""
        ...

class CHOP(OP):
    """Channel operator"""
    numSamples: int
    numChans: int
    chans: List['Channel']
    
    def clear(self) -> None:
        """Clear all channels"""
        ...
    
    def appendChan(self, name: str) -> 'Channel':
        """Append new channel"""
        ...

class Channel:
    """CHOP channel"""
    name: str
    index: int
    owner: CHOP
    
    def __getitem__(self, index: int) -> float:
        """Get sample value"""
        ...
    
    def __setitem__(self, index: int, value: float) -> None:
        """Set sample value"""
        ...

class scriptOP(CHOP):
    """Script CHOP operator"""
    inputs: List[OP]
    par: 'ParCollection'
    time: 'TimeCOMP'
    
    def appendChan(self, name: str) -> Channel:
        ...
    
    def clear(self) -> None:
        ...
    
    def __getitem__(self, key: str) -> Channel:
        ...

class scriptTOP(TOP):
    """Script TOP operator"""
    inputs: List[OP]
    par: 'ParCollection'
    
    def copyNumpyArray(self, array: Any) -> None:
        ...

class Par:
    """Parameter"""
    name: str
    label: str
    val: Any
    default: Any
    min: float
    max: float
    owner: OP
    
    def eval(self) -> Any:
        """Evaluate parameter value"""
        ...

class ParCollection:
    """Collection of parameters"""
    def __getattr__(self, name: str) -> Par:
        ...

class TimeCOMP:
    """Time component"""
    frame: float
    seconds: float
    rate: float

# Global functions
def op(path: Union[str, OP]) -> Optional[OP]:
    """Get operator by path"""
    ...

# Global objects
me: scriptOP
parent: Optional[OP]

# Type aliases for common TD objects
ScriptOp = Union[scriptOP, scriptTOP]