'''
Created on Jun 4, 2021

@author: Miguel
'''

import os
from datetime import datetime
import traceback
import xml.etree.ElementTree as ET
from xml.dom import minidom

def castIterable2Str(iterable, n=0):
    """ Given a dictionary, list or set in python, returns a well tabulated and 
    well formed to be printed (f.e: in Logs). 
    NOTE: Not designed for very large sized containers. 
    """
    tab_ = ".  ".join(['']*(n+1))
    tab2_ = ".  ".join(['']*(n+2))
    
    if isinstance(iterable, (list, set)):
        auxstr_ = '\n'+tab_+'['+'\n'
        for _val in iterable:
            auxstr_ += (tab2_+ castIterable2Str(_val, n+1)+',\n')
        if len(iterable)>0:
            auxstr_ = auxstr_[:-2] +'\n'
        return (auxstr_+tab_+']')
    
    if isinstance(iterable, dict):
        auxstr_ = '\n'+tab_+'{'+'\n'
        for _k, _val in iterable.items():
            auxstr_ += (tab2_+str(_k)+' : '+castIterable2Str(_val, n+1)+',\n')
        if len(iterable)>0:
            auxstr_ = auxstr_[:-2] +'\n'
        return auxstr_+tab_+'}'
    else:
        return str(iterable)

def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

class TextLines(object):
    """ Especial object to manage long string addition. """
    def __init__(self, html=False):
        """ string enumerable container
        :param html=False : value False does use '\n' as delimiter in the joining
                Use as True when parsing html line composition to """
        
        self.__lines  = dict()
        self.__length = 0
        
        self.__delimiter = '' if html else '\n'
        
    def join(self):
        str_ = self.__delimiter.join(self.__lines.values())
        return str_
    
    def append(self, addend):
        if isinstance(addend, (dict, list, set)):
            addend = castIterable2Str(addend)
        self.__lines[self.__length] = addend
        self.__length += 1
    
    def getLastLine(self):
        return self.__lines[self.__length-1] + '\n'

_line1 = "================================================================================\n"
_line2 = "\n--------------------------------------------------------------------------------\n"

dirPath = ''#os.getcwd()
if os.getcwd().startswith('/home'):
    dirPath = '/'.join(os.getcwd().split('\\'))

class Log(object):
    '''
    classdocs:
    '''
    __instance = None
    _DEBUG_FOLDER = 'results/_logs'
    #LEVELS: ERROR, WARN, INFO, DEBUG - default: INFO

    ## Use this level when logging information related to errors for which the execution must stop
    ERROR   = 'ERROR'
    WARNING = 'WARNING'
    DEBUG   = 'DEBBUG'
    SUMMARY = 'SUMMARY'
    
    def __init__(self):
        self.__debugFileName    = self._DEBUG_FOLDER + "/logsDebug.txt"
        self.__summaryFileName  = self._DEBUG_FOLDER + "/executionSummary.txt"

        #libPath = "/".join(os.path.realpath(__file__).split('\\')[:-2]) # also valid (remove /helper)
        libPath = "/".join(__file__.split("\\")[:-2]) if not dirPath else dirPath
        
        self.__debugFileName    = libPath + self.__debugFileName
        self.__summaryFileName  = libPath + self.__summaryFileName
        # Object ot register in order the messages for debug and 
        _header = self.__getDateHeader(startHeader=True)
        
        self._debugLogs   = TextLines()
        self._debugLogs.append(_header)
        self._summaryLogs = TextLines()
        self._summaryLogs.append(_header)
        
        self.__timers = {}
        
        self._generateFiles(create=True)
    
    
    def _write(self, message, level=None, raiseExc=None):
        st = datetime.now().strftime("%m-%d %H:%M:%S") 
        #st = datetime.datetime.now()
        msg_debug = '[{0}\t{1}] :: {2}'
        
        if level in (Log.WARNING, Log.ERROR):
            msg_debug = msg_debug.format(level, st, message)
            self._debugLogs.append(msg_debug)
            self._summaryLogs.append(msg_debug) # All incidences will be shown in             
            self._writeInFiles(both=True)
            if (level==Log.ERROR) and (raiseExc is not None):
                if isinstance(raiseExc(), (Exception, BaseException)):
                    Log.writeTraceback(raiseExc().__class__.__name__)
                    raise raiseExc(message)
        
        elif level in (Log.DEBUG, Log.SUMMARY):
            msg_debug = msg_debug.format(Log.DEBUG, st, message)
            self._debugLogs.append(msg_debug)
            if level == Log.SUMMARY:
                self._summaryLogs.append(message)
            self._writeInFiles(both=(level==Log.SUMMARY))
        
    
    @staticmethod
    def write(message, level=None, raiseExc=None):
        '''Adds a new Log entry:
        * Args:
        :param message: \a String representing the message to be written in the logs
        :param level=None: \a Enum representing the level of severity/verbosity. By default: DEBUG
        (param channel is not necessary in these tests)
        :param raiseExc=None: \a <Exception> or derived class to raise exception in case of ERROR
        * Example:
        Log.write('Destroyng tmpPageXmlDocs object', Log.DEBUG)
        --> DEBUG:: [2015-04-21 18:08:39] - Destroyng tmpPageXmlDocs object
        '''
        level = level if level else Log.DEBUG
        if not isinstance(level, (tuple, list)):
            level = tuple([level])
        
        log = Log.getInstance()
        for lev in level:
            log._write(message, lev, raiseExc)
        print(message)
            
    @staticmethod
    def writeTraceback(exception_type=''):
        """ Log the trace of any error that happened before a point in the code.
            Call this function in an exception block for a general exception or 
            from Log.write( , Log.ERROR, ).
        """
        self = Log.getInstance()
        
        _trace = traceback.extract_stack()
        # iterate the traces for a short message
        _traceback_str = []
        for _tr in _trace:
            _0 = _tr.__getattribute__('name')
            _1 = _tr.__getattribute__('lineno')
            _2 = _tr.__getattribute__('line')
            
            if 'log._write(' in _2:
                break
            _traceback_str.append(" > in [{}], ln[{}] :: {}".format(_0, _1, _2))
        
        try:
            _trace = traceback.format_exc()
            if exception_type != '':
                self._debugLogs.append('\nException type: <{}>'.format(exception_type))
                self._summaryLogs.append('\nException type: <{}>'.format(exception_type))
                
                self._debugLogs.append('\n'.join(_traceback_str))
                self._summaryLogs.append(_traceback_str[-1])
                
                self._writeInFiles(both=True)
            else:
                Log.write('\nException type: <{}>'.format(exception_type), 
                          Log.DEBUG)
            if _trace.strip() != 'NoneType: None':
                Log.write('\n'+_trace, Log.DEBUG)
        except UnicodeDecodeError as e:
            Log.write(str(e)+'\n\t(This error occurs when there are non utf-8 '
                      'characters in the last called module)', Log.WARNING)
    
    def _timeMethodConsumption(self, time, method_name):
        """ if the method is registered, add a call-count and append the time"""
        if method_name in self.__timers:
            self.__timers[method_name][0] += 1
            self.__timers[method_name][1] += time
        else:
            self.__timers[method_name] = [1, time]            
    
    @staticmethod
    def timeMethodConsumption(time, method_name):
        """
        :time must be float (use time.time()[1]-time.time()[0]) 
        :method_name string"""
        log = Log.getInstance()
        log._timeMethodConsumption(time, method_name)
        
    
    @staticmethod
    def getInstance():
        if Log.__instance == None:
            Log.__instance = Log()
        return Log.__instance
    
    @staticmethod
    def generateFiles():
        """ method to call form outside to close the logs"""
        self = Log.getInstance()
        self._generateFiles()
    
    def __getDateHeader(self, startHeader=None, endHeader=None, line=_line1):
        date_ = datetime.now()
        date_str = ''
        if startHeader:
            date_str = "\t\tExecuted on {} at {}\n".format(date_.date(), date_.time())
            return line + date_str
        elif endHeader:
            date_str = "\t\tExecution finished on {} at {}\n".format(date_.date(), date_.time())
            return date_str + line
            
        return line + date_str
        
    def _generateFiles(self, create=False):
        _mode, startHeader, endHeader = 'a', False, True
        if create:
            _mode, startHeader, endHeader = 'a', True, False
        
        ## get a table of timing values
        if self.__timers:
            _timeDataStr = ["{1:6}\t{2:10.4f}\t{0}\n".format(name, vals[0], vals[1]) 
                            for name, vals in self.__timers.items()]
            _timeDataStr = '\n{0}\n\tTIMERS:\n{1}\n{0}'.format(_line2, ''.join(_timeDataStr))
            Log.write(_timeDataStr, Log.DEBUG)
        
        _header = self.__getDateHeader(startHeader, endHeader)
        self._debugLogs.append(_header)
        self._summaryLogs.append(_header)
        
        if not os.path.exists(self._DEBUG_FOLDER):
            #create report folder if does not exist
            os.makedirs(self._DEBUG_FOLDER)
            
        with open(self.__debugFileName, _mode) as df:
            df.write(_header)   #self._debugLogs.join())
        with open(self.__summaryFileName, _mode) as df:
            df.write(_header)   #self._summaryLogs.join())
        if not create:
            print("Log Files Generated in report folder: \n\t{} \n\t{}"
                    .format(self.__debugFileName, self.__summaryFileName))
    
    def _writeInFiles(self, both=False):
        """ Default writes only in logsDebug, else, both=True, writes also in 
        the Summary file"""
        with open(self.__debugFileName, 'a') as df:
            df.write(self._debugLogs.getLastLine())
        if both:
            with open(self.__summaryFileName, 'a') as df:
                df.write(self._summaryLogs.getLastLine())
        
        

#===============================================================================
# LOG XML TREE LINES
# Save lines and nest them in order of occurrence
#===============================================================================
class XLog(object):
    
    '''
    Save events in order of apparition, useful to track elements in nested series
    
    XLog.write('Log File')
    for x in X:
        ...
        XLog.write('KEY X', attr1=x.name, attr2="Hey")
        sum_ = 0
        for y in Y(x):
            ...
            if (condition):
                XLog.write('KEY Y', namey=y.key txt="ConditionTrue")
                sum += 1
        XLog.write('KEY X', attr3=sum_)
    
    
    logs: 
    <Log_File>
        <KEY_X attr1='a' attr2="Hey">
            <KEY_Y y='y1' txt="ConditionTrue"/> 
            <KEY_Y y='y3' txt="ConditionTrue"/>(skips y2 ...)
        <KEY_X/>
        <KEY_X attr1='b' attr2="Hey">
            <KEY_Y y='y2' txt="ConditionTrue"/> 
        <KEY_X/>
    <Log_File/>
    
    '''
    
    __instance = None
    _DEBUG_FOLDER = 'results/_logs'
    
    def __init__(self, title='Tree_Log', prompt=False):
        
        libPath = "/".join(__file__.split("\\")[:-2]) if not dirPath else dirPath
        self._DEBUG_FOLDER = libPath + '/' + self._DEBUG_FOLDER
        self.__debugFolder    = self._DEBUG_FOLDER
        if not os.path.exists(self._DEBUG_FOLDER):
            os.mkdir(self._DEBUG_FOLDER)
        
        #libPath = "/".join(__file__.split("\\")[:-2]) if not dirPath else dirPath
        #self.__debugFileName    = libPath + self.__debugFileName
        
        # Object ot register in order the messages for debug and 
        # _header = self.__getDateHeader(startHeader=True)
        assert prompt in (True, False), "[prompt] argument must be boolean"
        
        self._tag_list = []
        self._parent_xpath = None
        self._xpath     = '.'
        
        self._debugLogs = ET.Element(title)
        self.prompt = prompt
        
        XLog.__instance = self
    
    def _setXpath(self, tag=None, curr_elem=None, level=0):
        path_ =''
        if curr_elem == None:
            curr_elem = self._debugLogs
            path_ = '.'
        
        elem = list(curr_elem)
        
        if elem == []:
            return path_
        if elem[-1].tag != tag or tag==None:
            path_ += ''.join(['/' , elem[-1].tag, "[last()]", 
                              self._setXpath(tag, elem[-1], level+1)])
        else:
            path_ += '/{}[last()]'.format(tag) 
        
        if level==0:
            self._xpath = path_
            if tag == None:
                self._parent_xpath = '/'.join(path_.split('/')[:-1])
            else:
                self._parent_xpath = path_.replace('/'+tag+'[last()]', '')
        return path_
    
    @staticmethod
    def _cast2strAttribValues(attribs):
        
        for key_, val in attribs.items():
            if not isinstance(val, str):
                if isinstance(val, float):
                    if (abs(val) > 1e-2) and abs(val) < 1e+3:
                        val = "{:5.4f}".format(val)
                    elif abs(val) < 1e-15:
                        val = "0.0"
                    else:
                        val = "{:5.4e}".format(val)
                attribs[key_] = str(val)
        return attribs
        
    def _accessLastEntry(self, tag, **new_attrs):
        new_attrs = self._cast2strAttribValues(new_attrs)
        
        # leaves = self._debugLogs.findall(xpath)
        leaves = self._debugLogs.findall(self._xpath)
        leaf = leaves[-1]
        if tag == leaf.tag:
            new_keys = [k not in new_attrs for k in leaf.attrib]
            if not False in new_keys:
                # append new attributes
                for k, val in new_attrs.items():
                    leaf.attrib[k] = val
            else:
                new_vals = [leaf.attrib[k]==new_attrs[k] for k in new_attrs]
                if not False in new_vals:
                    leaf.attrib = new_attrs #overwrite attributes
                else:
                    # append element (repeated attribute with different value)
                    parent = self._debugLogs.find(self._parent_xpath)
                    _ = ET.SubElement(parent, tag, new_attrs)
        else:
            # there is no tag value in the current leaf, append element
            leaf.append(ET.Element(tag, new_attrs))
        
        # print(prettify(self._debugLogs))
        # _=0
    
    def _write(self, tag, **attribs):     
        
        if tag in self._tag_list:
            self._setXpath(tag)
            self._accessLastEntry(tag, **attribs)
        else: 
            self._tag_list.append(tag)
            
            self._setXpath()
            self._accessLastEntry(tag, **attribs)
    
    
    @staticmethod
    def getInstance(tag_title):
        if XLog.__instance == None:
            XLog.__instance = XLog(tag_title)
        return XLog.__instance
    
    @staticmethod
    def write(tag_title, **attribs):
        '''
        Adds a new Log entry:
        * Args:
        :param tag_title: \a For element tag, will write on the last written element
        :param attribs: \a attribs  
        '''
        
        log = XLog.getInstance(tag_title)
        log._write(tag_title, **attribs)
    
    @staticmethod
    def getLog(filename=None):
        self = XLog.getInstance(None)
        data = prettify(self._debugLogs)#ET.tostring(self._debugLogs, encoding="unicode")
        
        if filename == None:
            filename = self.__debugFolder + "/debug_me.xml"
        else:
            filename = self.__debugFolder + "/" + filename
        
        # with open(filename, 'w+') as f:
        #     f.write(data)
        f = open(filename, 'w+')
        f.write(data)
        #print(data)
    
    @staticmethod
    def resetLog():
        XLog.__instance = None

if __name__ == "__main__":
    
    
    #===========================================================================
    # TESTING CODE:
    # EXPECTED OUTPUT:__________________________________________________________
    #
    # <?xml version="1.0" ?>
    # <Log_Test>
    #   <Type k="direct">
    #     <X_iter i="0" even="True" sum_j="4.5">
    #       <Y_check y="2"/>
    #       <Y_check y="4" cond="j eq 4"/>
    #       <Y_check y="6"/>
    #       <Y_check y="8"/>
    #     </X_iter>
    #     <X_iter i="1" even="False" sum_j="3.0">
    #       <Y_check y="2"/>
    #       <Y_check y="4" cond="j eq 4"/>
    #       <Y_check y="6"/>
    #     </X_iter>
    #     <X_iter i="2" even="True" sum_j="1.5">
    #       <Y_check y="2"/>
    #       <Y_check y="4" cond="j eq 4"/>
    #     </X_iter>
    #     <X_iter i="3" even="False" sum_j="1.5">
    #       <Y_check y="2"/>
    #     </X_iter>
    #   </Type>
    #   <Type k="exchange">
    #     <X_iter i="0" even="True" sum_j="16.5">
    #       <Y_check y="2"/>
    #       <Y_check y="4" cond="j eq 4"/>
    #       <Y_check y="6"/>
    #       <Y_check y="8"/>
    #     </X_iter>
    #     <X_iter i="1" even="False" sum_j="11.0">
    #       <Y_check y="2"/>
    #       <Y_check y="4" cond="j eq 4"/>
    #       <Y_check y="6"/>
    #     </X_iter>
    #     <X_iter i="2" even="True" sum_j="5.5">
    #       <Y_check y="2"/>
    #       <Y_check y="4" cond="j eq 4"/>
    #     </X_iter>
    #     <X_iter i="3" even="False" sum_j="5.5">
    #       <Y_check y="2"/>
    #     </X_iter>
    #   </Type>
    # </Log_Test>
    #
    #===========================================================================
    
    XLog('Log_Test')
    for k in ('direct', 'exchange'):
        if k == 'exchange':
            _=0
        XLog.write('Type', k=k)
        for i in range(4):
            
            XLog.write('X_iter', i = i, even = bool(i%2 == 0))
            sum_ = 0
            for j in range(2, 10 - 2*i, 2):
                XLog.write('Y_check', y = j)
                if j != 4:
                    if k == 'exchange':
                        sum_ += 5.5
                    else:
                        sum_ += 1.5
                else:
                    XLog.write('Y_check', cond='j eq 4')
            XLog.write('X_iter', sum_j= sum_)
    
    XLog.getLog()
        
        