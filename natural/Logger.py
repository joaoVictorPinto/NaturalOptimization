# Trabalho de CPE 726 - Topicos Especiais em Inferencia em Grafos
# Author: Joao Victor da Fonseca Pinto

__all__ = ['EnumStringification', 'LoggingLevel', 'Logger','retrieve_kw', 'NotSet']

import logging
#from RingerCore import os_environ_get
# TODO With the Configure method it is possible to create a singleton to
# retrieve all loggers using the global configured message level

class NotSetType( type ):
  def __bool__(self):
    return False
  __nonzero__ = __bool__
  def __repr__(self):
    return "<+NotSet+>"
  def __str__(self):
    return "<+NotSet+>"

class NotSet( object ): 
  """As None, but can be used with retrieve_kw to have a unique default value
  though all job hierarchy."""
  __metaclass__ = NotSetType


def retrieve_kw( kw, key, default = NotSet ):
  """
  Use together with NotSet to have only one default value for your job
  properties.
  """
  if not key in kw or kw[key] is NotSet:
    kw[key] = default
  return kw.pop(key)



class EnumStringification( object ):
  "Adds 'enum' static methods for conversion to/from string"

  _ignoreCase = False

  @classmethod
  def tostring(cls, val):
    "Transforms val into string."
    for k, v in get_attributes(cls, getProtected = False):
      if v==val:
        return k
    return None

  @classmethod
  def fromstring(cls, str_):
    "Transforms string into enumeration."
    if not cls._ignoreCase:
      return getattr(cls, str_, None)
    else:
      allowedValues = [attr for attr in get_attributes(cls) if not attr[0].startswith('_')]
      try:
        idx = [attr[0].upper() for attr in allowedValues].index(str_.upper().replace('-','_'))
      except ValueError:
        raise ValueError("%s is not in enumeration. Use one of the followings: %r" % (str_, allowedValues) )
      return allowedValues[idx][1]

  @classmethod
  def retrieve(cls, val):
    """
    Retrieve int value and check if it is a valid enumeration string or int on
    this enumeration class.
    """
    allowedValues = [attr for attr in get_attributes(cls) if not attr[0].startswith('_')]
    try:
      # Convert integer string values to integer, if possible:
      val = int(val)
    except ValueError:
      pass
    if type(val) is str:
      oldVal = val
      val = cls.fromstring(val)
      if val is None:
          raise ValueError("String (%s) does not match any of the allowed values %r." % \
              (oldVal, allowedValues))
    else:
      if not val in [attr[1] for attr in allowedValues]:
        raise ValueError(("Attempted to retrieve val benchmark "
            "with a enumeration value which is not allowed. Use one of the followings: "
            "%r") % allowedValues)
    return val

  @classmethod
  def stringList(cls):
    from operator import itemgetter
    return [v[0] for v in sorted(get_attributes( cls, getProtected = False), key=itemgetter(1))]

  @classmethod
  def intList(cls):
    from operator import itemgetter
    return [v[1] for v in sorted(get_attributes( cls, getProtected = False), key=itemgetter(1))]

class FatalError(RuntimeError):
  pass


class LoggingLevel ( EnumStringification ):
  """
    A wrapper for logging levels, which allows stringification of known log
    levels.
  """
  VERBOSE  = logging.DEBUG - 1
  DEBUG    = logging.DEBUG
  INFO     = logging.INFO
  WARNING  = logging.WARNING
  ERROR    = logging.ERROR
  CRITICAL = logging.CRITICAL
  FATAL    = logging.CRITICAL
  MUTE     = logging.CRITICAL # MUTE Still displays fatal messages.

  @classmethod
  def toC(cls, val):
    val = LoggingLevel.retrieve( val ) 
    if val == cls.VERBOSE:
      val = 0
    else:
      val = val/10
    return val + 1 # There is NIL at 0, DEBUG is 2 and so on.

logging.addLevelName(LoggingLevel.VERBOSE, "VERBOSE")
logging.addLevelName(LoggingLevel.FATAL,    "FATAL" )

def verbose(self, message, *args, **kws):
  """
    Attempt to emit verbose message
  """
  if self.isEnabledFor(LoggingLevel.VERBOSE):
    self._log(LoggingLevel.VERBOSE, message, args, **kws) 


def _getAnyException(args):
  exceptionType = [issubclass(arg,BaseException) if type(arg) is type else False for arg in args]
  Exc = None
  if any(exceptionType):
    # Check if any args message is the exception type that should be raised
    args = list(args)
    Exc = args.pop( exceptionType.index( True ) )
    args = tuple(args)
  return Exc, args

def warning(self, message, *args, **kws):
  Exc, args = _getAnyException(args)
  if self.isEnabledFor(LoggingLevel.WARNING):
    self._log(LoggingLevel.WARNING, message, args, **kws) 
  if Exc is not None:
    if args:
      raise Exc(message % (args if len(args) > 1 else args[0]))
    else:
      raise Exc(message)

def error(self, message, *args, **kws):
  Exc, args = _getAnyException(args)
  if self.isEnabledFor(LoggingLevel.ERROR):
    self._log(LoggingLevel.ERROR, message, args, **kws) 
  if Exc is not None:
    if args:
      raise Exc(message % (args if len(args) > 1 else args[0]))
    else:
      raise Exc(message)

def fatal(self, message, *args, **kws):
  """
    Attempt to emit fatal message
  """
  Exc, args = _getAnyException(args)
  if Exc is None: Exc = FatalError
  if self.isEnabledFor(LoggingLevel.FATAL):
    self._log(LoggingLevel.FATAL, message, args, **kws) 
  if args:
    raise Exc(message % (args if len(args) > 1 else args[0]))
  else:
    raise Exc(message)

logging.Logger.verbose = verbose
logging.Logger.warning = warning
logging.Logger.error = error
logging.Logger.fatal = fatal
logging.Logger.critical = fatal





# The logger main object
class Logger(object):

  """
    >>> Internal method to get the formatter custom obj.
  """
  def _getFormatter(self):
    class Formatter(logging.Formatter):
      import numpy as np
      gray, red, green, yellow, blue, magenta, cyan, white = ['0;%d' % int(d) for d in (30 + np.arange(8))]
      bold_black, bold_red, bold_green, bold_yellow, bold_blue, bold_magenta, bold_cyan, \
          bold_white = ['1;%d' % d for d in 90 + np.arange(8)]
      gray = '1;30'
      reset_seq = "\033[0m"
      color_seq = "\033[%(color)sm"
      colors = {
                 'VERBOSE':  gray,
                 'DEBUG':    cyan,
                 'INFO':     green,
                 'WARNING':  bold_yellow,
                 'ERROR':    red,
                 'CRITICAL': bold_red,
                 'FATAL':    bold_red,
               }
  
      # It's possible to overwrite the message color by doing:
      # logger.info('MSG IN MAGENTA', extra={'color' : Logger._formatter.bold_magenta})
  
      def __init__(self, msg, use_color = False):
        if use_color:
          logging.Formatter.__init__(self, self.color_seq + msg + self.reset_seq )
        else:
          logging.Formatter.__init__(self, msg)
        self.use_color = use_color
  
      def format(self, record):
        if not(hasattr(record,'nl')):
          record.nl = True
        levelname = record.levelname
        if not 'color' in record.__dict__ and self.use_color and levelname in self.colors:
          record.color = self.colors[levelname]
        return logging.Formatter.format(self, record)
  
    import os, sys
    formatter = Formatter(
                         "Py.%(name)-33.33s %(levelname)7.7s %(message)s", 
                         #not(int(os.environ.get('RCM_NO_COLOR',1)) or not(sys.stdout.isatty()))
                         True
                         )
    return formatter
  

  def __init__(self, **kw):
    import sys
    # create logger with 'spam_application'
    from copy import copy
    self._level = retrieve_kw( kw, 'level', LoggingLevel.INFO)
    self._logger = logging.getLogger(self.__class__.__name__)
    ch = logging.StreamHandler(sys.__stdout__)
    formatter =  self._getFormatter()
    ch.setLevel(logging.NOTSET)
    ch.setFormatter(formatter)
    # add the handlers to #the logger
    self._logger.handlers = [] # Force only one handler
    self._logger.addHandler(ch)
    self._logger.setLevel(self._level)

  def setLevel(self, lvl):
    self._logger.setLevel(lvl)


  def getLevel(self):
    return self._level

  def getModuleLogger(self):
    return self._logger

