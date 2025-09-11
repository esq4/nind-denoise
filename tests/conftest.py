import types
import pytest

class _FakeExifImage:
    def __init__(self):
        self._data = {'Exif.Image.Make': 'UnitTest'}
    def readMetadata(self):
        return None
    def exifData(self):
        return self._data
    def setExifData(self, data):
        self._data = data
    def writeMetadata(self):
        return None

class _FakeImageFactory:
    def __init__(self):
        self._last = _FakeExifImage()
    def open(self, path):
        return _FakeExifImage()

@pytest.fixture
def fake_exiv2_module():
    fake = types.SimpleNamespace(ImageFactory=_FakeImageFactory())
    return fake

@pytest.fixture
def sample_xmp(tmp_path):
    # minimal XMP with a few operations
    content = (
        "<?xml version='1.0' encoding='UTF-8'?>\n"
        "<x:xmpmeta xmlns:x='adobe:ns:meta/' xmlns:darktable='http://darktable.sf.net/' xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#'>\n"
        "  <rdf:RDF>\n"
        "    <rdf:Description darktable:iop_order_version='3' darktable:iop_order_list='demosaic,0,colorin,0,exposure,0,toneequal,0,flip,0'>\n"
        "      <darktable:history>\n"
        "        <rdf:li darktable:num='1' darktable:operation='demosaic' darktable:enabled='1'/>\n"
        "        <rdf:li darktable:num='2' darktable:operation='colorin' darktable:enabled='1'/>\n"
        "        <rdf:li darktable:num='3' darktable:operation='exposure' darktable:enabled='1'/>\n"
        "        <rdf:li darktable:num='4' darktable:operation='toneequal' darktable:enabled='1'/>\n"
        "        <rdf:li darktable:num='5' darktable:operation='flip' darktable:enabled='1'/>\n"
        "        <rdf:li darktable:num='6' darktable:operation='unlistedop' darktable:enabled='1'/>\n"
        "      </darktable:history>\n"
        "    </rdf:Description>\n"
        "  </rdf:RDF>\n"
        "</x:xmpmeta>\n"
    )
    xmp = tmp_path / 'img.RAF.xmp'
    xmp.write_text(content, encoding='utf-8')
    return xmp
