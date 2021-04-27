# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 04:06:28 2020

@author: olivar
"""

import io
import os
import sys
from PyQt5 import uic, QtGui, QtCore, QtWidgets
import obspy
import pickle
from functools import partial
#import isp.receiverfunctions.rf_dialogs_utils as du
import rfun_dialogs_utils as du
#from isp.Gui.Frames import UiReceiverFunctionsCut, UiReceiverFunctionsSaveFigure, UiReceiverFunctionsCrossSection, BaseFrame
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
#from isp.Gui import pyqt, pqg, pw, pyc, qt
from PyQt5 import uic, QtGui, QtCore, QtWidgets
import numpy as np

class ShowEarthquakeDialog(QtWidgets.QDialog):
    def __init__(self, file, bandpass):
        super(ShowEarthquakeDialog, self).__init__()
        uic.loadUi('ui/RfunDialogsShowEarthquake.ui', self)
        
        # Button connections
        self.pushButton_2.clicked.connect(self.close)
        
        self.mplwidget.figure.subplots(nrows=3)
        st = obspy.read(file)
        st.filter('bandpass', freqmin=bandpass[0], freqmax=bandpass[1])
        
        try:
            l = st.select(component="L")
            maxy = np.max(l[0].data)
            miny = np.min(l[0].data)
        except IndexError: # If this happens there is no L component, so there should be a Z component
            z = st.select(component="Z")
            maxy = np.max(z[0].data)
            miny = np.min(z[0].data)
        
        for i, tr in enumerate(st):
            self.mplwidget.figure.axes[i].plot(tr.times("matplotlib"), tr.data, color="black")
            self.mplwidget.figure.axes[i].set_ylim(miny, maxy)
    
    def close(self):
        self.done(0)

class CutEarthquakesDialog(QtWidgets.QDialog):
    def __init__(self):
        super(CutEarthquakesDialog, self).__init__()
        uic.loadUi('ui/RfunDialogsCutEqs.ui', self)
        self.show()

        # connectionsx
        self.pushButton_2.clicked.connect(partial(self.get_path, 2))
        self.pushButton_3.clicked.connect(partial(self.get_path, 3))
        self.pushButton_4.clicked.connect(partial(self.get_path, 4))
        self.pushButton_5.clicked.connect(partial(self.get_path, 5))        
        self.pushButton_6.clicked.connect(self.cut_earthquakes)
        self.pushButton_7.clicked.connect(self.close)
    
    def get_path(self, pushButton):
        if pushButton == 2:
            path = QtWidgets.QFileDialog.getExistingDirectory()
            self.lineEdit.setText(path)
        elif pushButton == 3:
            path = QtWidgets.QFileDialog.getOpenFileName()[0]
            self.lineEdit_3.setText(path)
        elif pushButton == 4:
            path = QtWidgets.QFileDialog.getExistingDirectory()
            self.lineEdit_2.setText(path)
        elif pushButton == 5:
            path = QtWidgets.QFileDialog.getSaveFileName()[0]
            self.lineEdit_4.setText(path)

    def cut_earthquakes(self):
        data_path = self.lineEdit.text()
        station_metadata_path = self.lineEdit_3.text()
        earthquake_output_path = self.lineEdit_2.text()
        event_metadata_output_path = self.lineEdit_4.text()
        starttime = self.dateTimeEdit.dateTime().toString("yyyy-MM-ddThh:mm:ss.zzz000Z")
        endtime = self.dateTimeEdit_2.dateTime().toString("yyyy-MM-ddThh:mm:ss.zzz000Z")
        min_mag = self.doubleSpinBox_2.value()
        min_snr = self.doubleSpinBox.value()
        min_dist = self.doubleSpinBox_3.value()
        max_dist = self.doubleSpinBox_4.value()
        client = self.comboBox.currentText()
        model = self.comboBox_2.currentText()
        
        print(starttime)
        print(endtime)
        
        catalog = du.get_catalog(starttime, endtime, client=client, min_magnitude=min_mag)
        
        arrivals = du.taup_arrival_times(catalog, station_metadata_path, earth_model=model,
                                            min_distance_degrees=min_dist,
                                            max_distance_degrees=max_dist)
        pickle.dump(arrivals, open(event_metadata_output_path, "wb"))
        
        if self.checkBox.isChecked():
            data_map = du.map_data(data_path, quick=True)
        else:
            data_map = du.map_data(data_path, quick=False)
        
        time_before = self.doubleSpinBox_5.value()
        time_after = self.doubleSpinBox_6.value()
        rotation = self.comboBox_3.currentText()
        remove_instrumental_responses = self.checkBox_2.isChecked()

        du.cut_earthquakes(data_map, arrivals, time_before, time_after, min_snr,
                    station_metadata_path, earthquake_output_path)

class SaveFigureDialog(QtWidgets.QDialog):
    def __init__(self, figure, preferred_size, preferred_margins, preferred_title,
                 preferred_xlabel, preferred_ylabel, preferred_fname):
        super(SaveFigureDialog, self).__init__()
        uic.loadUi('ui/RfunDialogsSaveFigure.ui', self)
        
        self.figure = figure
        
        self.pushButton_2.clicked.connect(self.save_figure)
        
        self.doubleSpinBox.setValue(preferred_size[0])
        self.doubleSpinBox_2.setValue(preferred_size[1])
        self.doubleSpinBox_3.setValue(preferred_margins[0])
        self.doubleSpinBox_4.setValue(preferred_margins[1])
        self.doubleSpinBox_6.setValue(preferred_margins[2])
        self.doubleSpinBox_5.setValue(preferred_margins[3])
        self.lineEdit.setText(preferred_title)
        self.lineEdit_2.setText(preferred_xlabel)
        self.lineEdit_3.setText(preferred_ylabel)
        
        self.preferred_fname = preferred_fname
    
    def save_figure(self):
        output_path = QtWidgets.QFileDialog.getSaveFileName(directory=self.preferred_fname)[0]
        
        # Apply settings
        self.figure.set_size_inches(self.doubleSpinBox_2.value(), self.doubleSpinBox.value())
        self.figure.suptitle(self.lineEdit.text())
        self.figure.axes[-1].set_xlabel(self.lineEdit_2.text())
        self.figure.axes[-1].set_ylabel(self.lineEdit_3.text())
        self.figure.subplots_adjust(left=self.doubleSpinBox_3.value(), bottom=self.doubleSpinBox_6.value(),
                                    right=self.doubleSpinBox_4.value(), top=self.doubleSpinBox_5.value())
        
        format_ = self.comboBox.currentText()         
        self.figure.savefig(output_path + format_, dpi=self.spinBox.value(), format=format_[1:])

class CrossSectionDialog(QtWidgets.QDialog):
    def __init__(self, x, y, z, start, end):
        super(CrossSectionDialog, self).__init__()
        uic.loadUi('ui/RfunDialogsCrossSection.ui', self)
        
        self.start = start
        self.end = end
        self.x = x
        self.y = y
        self.z = z
        
        self.mplwidget.figure.subplots(1)
        im = self.mplwidget.figure.axes[0].pcolormesh(x, y, z, vmin=np.min(z), vmax=np.max(z), cmap="bwr")
        
        divider = make_axes_locatable(self.mplwidget.figure.axes[0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        self.mplwidget.figure.axes[0].set_xlabel("Distance (km)")
        self.mplwidget.figure.axes[0].set_ylabel("Depth (km)")
        
        self.pushButton.clicked.connect(self.save_cross_section)
        self.pushButton_2.clicked.connect(self.save_figure)
        self.pushButton_3.clicked.connect(self.close)
    
    def save_cross_section(self):
        fname = QtWidgets.QFileDialog.getSaveFileName()[0]
        
        css_dict = {"start":self.start,
                    "end":self.end,
                    "distance":self.x,
                    "depth":self.y,
                    "cross_section":self.z}

        if fname:
            pickle.dump(css_dict, open(fname, "wb"))
    
    def save_figure(self):
        buffer = io.BytesIO()
        pickle.dump(self.mplwidget.figure, buffer)
        # Point to the first byte of the buffer and read it
        buffer.seek(0)
        fig_copy = pickle.load(buffer)
        
        #We also need a new canvas manager
        newfig = plt.figure()
        newmanager = newfig.canvas.manager
        newmanager.canvas.figure = fig_copy
        fig_copy.set_canvas(newmanager.canvas)        
        
        dialog = SaveFigureDialog(fig_copy, preferred_size, preferred_margins, preferred_title,
                 preferred_xlabel, preferred_ylabel, preferred_fname)
        dialog.exec_()
    
    def close(self):
        self.done(0)