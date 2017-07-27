from __future__ import absolute_import
import requests


class XNATError(Exception):
    pass


class XNATValueError(XNATError, ValueError):
    pass


class XNATResponseError(XNATValueError):
    pass


class XNATIOError(XNATError, IOError):
    pass


class XNATUploadError(XNATIOError):
    pass


class XNATSSLError(XNATError, requests.exceptions.SSLError):
    pass


