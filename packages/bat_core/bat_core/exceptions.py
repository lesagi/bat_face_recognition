class BatCoreError(Exception):
    pass


class InvalidManifestError(BatCoreError):
    pass


class InterfaceViolationError(BatCoreError):
    pass
