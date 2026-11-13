class ApplicationError(Exception):
    """Base para erros conhecidos da aplicacao."""


class UnknownSegmentationMethod(ApplicationError, ValueError):
    pass


class InvalidSegmentationMask(ApplicationError, ValueError):
    pass


class UnsafeSegmentation(ApplicationError, ValueError):
    pass


class InvalidCommand(ApplicationError, ValueError):
    pass

