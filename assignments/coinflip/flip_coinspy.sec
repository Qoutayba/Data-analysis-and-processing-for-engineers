#!/usr/bin/env python3
import numpy as np
import sys
import os
import h5py 
import time
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QApplication, QMessageBox
from PyQt5.QtCore import pyqtSlot, pyqtBoundSignal, pyqtSignal, QThreadPool


##### Frame templates  generated via pyuic5 
class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(365, 136)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.open_toss = QtWidgets.QPushButton(self.centralwidget)
        self.open_toss.setGeometry(QtCore.QRect(240, 10, 121, 21))
        self.open_toss.setObjectName("open_toss")
        self.new_set = QtWidgets.QRadioButton(self.centralwidget)
        self.new_set.setGeometry(QtCore.QRect(20, 10, 161, 23))
        self.new_set.setChecked(True)
        self.new_set.setObjectName("new_set")
        self.existing_set = QtWidgets.QRadioButton(self.centralwidget)
        self.existing_set.setGeometry(QtCore.QRect(20, 40, 161, 23))
        self.existing_set.setChecked(False)
        self.existing_set.setObjectName("existing_set")
        self.quit = QtWidgets.QPushButton(self.centralwidget)
        self.quit.setGeometry(QtCore.QRect(240, 40, 121, 21))
        self.quit.setToolTip("")
        self.quit.setObjectName("quit")
        self.save_location = QtWidgets.QLabel(self.centralwidget)
        self.save_location.setGeometry(QtCore.QRect(20, 70, 341, 17))
        self.save_location.setObjectName("save_location")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 365, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Main Window"))
        self.open_toss.setToolTip(_translate("MainWindow", "<html><head/><body><p>to your witcher.</p></body></html>"))
        self.open_toss.setText(_translate("MainWindow", "Toss a coin"))
        self.new_set.setText(_translate("MainWindow", "Create new set"))
        self.existing_set.setText(_translate("MainWindow", "Add to existing set"))
        self.quit.setText(_translate("MainWindow", "Quit"))
        self.save_location.setText(_translate("MainWindow", "TextLabel"))


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(356, 100)
        self.label = QtWidgets.QLabel(Form)
        self.label.setGeometry(QtCore.QRect(10, 10, 171, 17))
        self.label.setObjectName("label")
        self.current = QtWidgets.QLabel(Form)
        self.current.setGeometry(QtCore.QRect(180, 10, 31, 17))
        self.current.setObjectName("current")
        self.prev_label = QtWidgets.QLabel(Form)
        self.prev_label.setGeometry(QtCore.QRect(10, 30, 171, 17))
        self.prev_label.setObjectName("prev_label")
        self.previous = QtWidgets.QLabel(Form)
        self.previous.setGeometry(QtCore.QRect(100, 30, 31, 17))
        self.previous.setObjectName("previous")
        self.disable_box = QtWidgets.QGroupBox(Form)
        self.disable_box.setGeometry(QtCore.QRect(0, -20, 361, 121))
        self.disable_box.setTitle("")
        self.disable_box.setObjectName("disable_box")
        self.tails = QtWidgets.QPushButton(self.disable_box)
        self.tails.setGeometry(QtCore.QRect(120, 90, 89, 25))
        self.tails.setObjectName("tails")
        self.heads = QtWidgets.QPushButton(self.disable_box)
        self.heads.setGeometry(QtCore.QRect(10, 90, 89, 25))
        self.heads.setObjectName("heads")
        self.quit = QtWidgets.QPushButton(self.disable_box)
        self.quit.setGeometry(QtCore.QRect(238, 30, 111, 25))
        self.quit.setObjectName("quit")

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.label.setText(_translate("Form", "Total number of tosses:"))
        self.current.setText(_translate("Form", "0"))
        self.prev_label.setText(_translate("Form", "Previously: "))
        self.previous.setText(_translate("Form", "0"))
        self.tails.setText(_translate("Form", "Tails"))
        self.heads.setText(_translate("Form", "Heads"))
        self.quit.setText(_translate("Form", "Save and Quit"))


################## Main frame which takes you to the toss interface ####################################
class MainWindow( QMainWindow, Ui_MainWindow):
    def __init__( self, parent=None):
        super( MainWindow, self).__init__( parent)
        self.setupUi( self)
        self.h5fname =  'coin_tosses.h5'
        self.save_location.setText( 'Results will be saved to: "{}"'.format( self.h5fname) )
        if os.path.isfile( self.h5fname):
            self.existing_set.setChecked( True)
            self.new_set.setChecked( False)

    @pyqtSlot( bool)
    def on_existing_set_toggled( self, state):
        if not os.path.isfile( self.h5fname) and state:
            """ give feedback that the file does not exist"""
            reply = QMessageBox.warning( self,
                "Warning", "File of previous results does not exist.",
            QMessageBox.Ok ) 
            self.existing_set.setChecked( not state)
            self.new_set.setChecked( state)
            return
        else:
            self.existing_set.setChecked( state)
            self.new_set.setChecked( not state)

    @pyqtSlot( bool)
    def on_new_set_toggled( self, state):
        if os.path.isfile( self.h5fname) and state:
            """ give feedback that the file does not exist"""
            reply = QMessageBox.warning( self,
                "Warning", "File of previous results will be deleted!",
            QMessageBox.Ok ) 
        self.new_set.setChecked( state)
        self.existing_set.setChecked( not state)

    @pyqtSlot()
    def on_open_toss_pressed( self):
        self.popup = Popup( self.new_set.isChecked(), self.h5fname, self)
        self.new_set.setChecked( False)
        self.existing_set.setChecked( True)
        self.popup.show()

    @pyqtSlot()
    def on_quit_pressed( self):
        sys.exit( 0)

##### popup where the data is actually collected ####
class Popup( QMainWindow, Ui_Form):
    def __init__( self, new_set, h5fname, parent=None):
        super( Popup, self).__init__( parent)
        self.setupUi( self)
        self.h5fname = h5fname
        if new_set:
            self.prev_label.hide()
            self.previous.hide()
            self.h5file = h5py.File( self.h5fname, 'w' )
        else:
            self.prev_label.show()
            self.previous.show()
            self.h5file = h5py.File( self.h5fname, 'a' )
            try:
                self.previous.setText( str(len( np.array(self.h5file['tosses']) ) ) )
            except:
                self.previous.setText( '0')
        self.counter = 0
        self.tosses = []
        self.delay = 1 #1 second delay between coin tosses
        self.pause_button = FunctionThread( self.delay)
        #self.pause_button.signal_handle.connect( self.pause )
        self.pause_button.disable.connect( self.pause )
        self.pause_button.enable.connect( self.unpause )


    def pause( self):
        self.disable_box.setDisabled( True)

    def unpause( self):
        self.disable_box.setDisabled( False)


    @pyqtSlot()
    def on_heads_clicked( self):
        self.counter += 1
        self.current.setText( str( self.counter) )
        self.tosses.append( 0 )
        self.pause_button.start()


    @pyqtSlot()
    def on_tails_clicked( self):
        self.counter += 1
        self.current.setText( str( self.counter) )
        self.tosses.append( 1 )
        self.pause_button.start()


    @pyqtSlot()
    def on_quit_pressed( self):
        if len( self.tosses) == 0:
            self.h5file.close()
            self.destroy()
            return
        if 'tosses' in self.h5file:
            x = list( self.h5file['tosses'][:] )
            x.extend( self.tosses)
            self.tosses = x
            del self.h5file['tosses']
        if len( self.tosses) == 1:
            tosses = np.array( [self.tosses] )[0]
        else:
            tosses = np.array(self.tosses)
        print( tosses.shape) 
        try:
            self.h5file.create_dataset( 'tosses', data=tosses, dtype='u1', compression='gzip')
        except:
            pass
        self.h5file.close()
        self.destroy()



## pyqt generator threads that the UI is updated in real time when clicking and 'resonding'
class FunctionThread( QtCore.QThread):#, QtCore.QRunnable):
    disable = QtCore.pyqtSignal()
    enable = QtCore.pyqtSignal()
    def __init__( self, delay=1):
        super( FunctionThread, self).__init__()
        self.delay = delay
    
    def run( self, *args, **kwargs):
        self.disable.emit( *args, **kwargs)
        time.sleep( self.delay)
        self.enable.emit()


if __name__ == '__main__':
    app = QApplication( sys.argv)
    ui  = MainWindow( )
    ui.show()
    sys.exit( app.exec_())

