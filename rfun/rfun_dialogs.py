# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 04:06:28 2020

@author: olivar
"""

import os
from PyQt5 import uic, QtWidgets
import obspy
import pickle
from functools import partial
import rfun.rfun_dialogs_utils as du
import rfun.rfun_main_window_utils as mwu

from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

from rfun.definitions import ROOT_DIR, CONFIG_PATH

class ShowEarthquakeDialog(QtWidgets.QDialog):
    def __init__(self, file, bandpass):
        super(ShowEarthquakeDialog, self).__init__()
        uic.loadUi(os.path.join(ROOT_DIR, 'ui/RfunDialogsShowEarthquake.ui'), self)
        
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
        uic.loadUi(os.path.join(ROOT_DIR, 'ui/RfunDialogsCutEqs.ui'), self)
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
            if path:
                self.lineEdit.setText(path)
        elif pushButton == 3:
            path = QtWidgets.QFileDialog.getOpenFileName()[0]
            if path:
                self.lineEdit_3.setText(path)
        elif pushButton == 4:
            path = QtWidgets.QFileDialog.getExistingDirectory()
            if path:
                self.lineEdit_2.setText(path)
        elif pushButton == 5:
            path = QtWidgets.QFileDialog.getSaveFileName()[0]
            if path:
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
        remove_response = self.checkBox_2.isChecked()

        du.cut_earthquakes(data_map, arrivals, time_before, time_after, min_snr,
                    station_metadata_path, earthquake_output_path, remove_response=remove_response)

class SaveFigureDialog(QtWidgets.QDialog):
    def __init__(self, figure, preferred_size, preferred_margins, preferred_title,
                 preferred_xlabel, preferred_ylabel, preferred_fname):
        super(SaveFigureDialog, self).__init__()
        uic.loadUi(os.path.join(ROOT_DIR, 'ui/RfunDialogsSaveFigure.ui'), self)
        
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
        
        if output_path:
        
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
        uic.loadUi(os.path.join(ROOT_DIR, 'ui/RfunDialogsCrossSection.ui'), self)
        
        self.start = start
        self.end = end
        self.x = x
        self.y = y
        self.z = z
        
        self.mplwidget.figure.subplots(1)
        im = self.mplwidget.figure.axes[0].pcolormesh(x, y, z, vmin=-1., vmax=1, cmap="RdBu_r")
        
        divider = make_axes_locatable(self.mplwidget.figure.axes[0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        self.mplwidget.figure.colorbar(im, cax=cax)
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
        fname = QtWidgets.QFileDialog.getSaveFileName(None, "Save figure", "", "PNG (*.png)")[0]
        if fname:
            self.mplwidget.figure.savefig(fname, dpi=600)
            
    def close(self):
        self.done(0)

class PreferencesDialog(QtWidgets.QDialog):
    def __init__(self):
        super(PreferencesDialog, self).__init__()
        uic.loadUi(os.path.join(ROOT_DIR, 'ui/RfunDialogsPreferences.ui'), self)
        
        self.pushButton_2.clicked.connect(self.save_settings)
        self.pushButton_3.clicked.connect(self.close)
        self.pushButton.clicked.connect(self.reset_to_defaults)
        
        self.settings = mwu.read_preferences()
        self.read_settings()
    
    def read_settings(self):
        
        # CCP stack settings
        # Appearance
        self.checkBox_6.setChecked(self.settings['ccp']['appearance']['include_stations'])
        self.comboBox_6.setCurrentIndex(self.comboBox_6.findText(self.settings['ccp']['appearance']['plotting_method']))
        self.comboBox_5.setCurrentIndex(self.comboBox_5.findText(self.settings['ccp']['appearance']['colormap']))
        self.lineEdit_14.setText(self.settings['ccp']['appearance']['station_marker'])
        self.lineEdit_13.setText(self.settings['ccp']['appearance']['station_marker_color'])
        # Shapefiles
        self.checkBox_5.setChecked(self.settings['ccp']['shapefiles']['include'])
        self.lineEdit_12.setText(self.settings['ccp']['shapefiles']['path'])
        # Computation
        self.comboBox_8.setCurrentIndex(self.comboBox_8.findText(self.settings['ccp']['computation']['stacking_method']))
        
        # RFS settings
        # Appearance
        self.lineEdit.setText(self.settings['rfs']['appearance']['line_color'])
        self.doubleSpinBox.setValue(self.settings['rfs']['appearance']['line_width'])
        self.lineEdit_3.setText(self.settings['rfs']['appearance']['positive_fill_color'])
        self.lineEdit_2.setText(self.settings['rfs']['appearance']['negative_fill_color'])
        # Computation
        self.checkBox.setChecked(self.settings['rfs']['computation']['normalize'])
        self.doubleSpinBox_2.setValue(self.settings['rfs']['computation']['w0'])
        self.doubleSpinBox_3.setValue(self.settings['rfs']['computation']['time_shift'])
        # Stacking
        self.doubleSpinBox_8.setValue(self.settings['rfs']['stacking']['ref_slowness'])
        
        # HK settings
        # Appearance
        self.comboBox.setCurrentIndex(self.comboBox.findText(self.settings['hk']['appearance']['plotting_method']))
        self.comboBox_2.setCurrentIndex(self.comboBox_2.findText(self.settings['hk']['appearance']['colormap']))
        self.lineEdit_7.setText(self.settings['hk']['appearance']['line_color'])
        self.lineEdit_8.setText(self.settings['hk']['appearance']['ser_color'])
        # Computation
        self.checkBox_2.setChecked(self.settings['hk']['computation']['semblance_weighting'])
        self.spinBox.setValue(self.settings['hk']['computation']['H_points'])
        self.spinBox_2.setValue(self.settings['hk']['computation']['k_points'])
        self.doubleSpinBox_5.setValue(self.settings['hk']['computation']['avg_vp'])
        # Theoretical arrival times
        self.doubleSpinBox_6.setValue(self.settings['hk']['theoretical_atimes']['ref_slowness'])
        self.doubleSpinBox_7.setValue(self.settings['hk']['theoretical_atimes']['avg_vp'])

        # Crustal thickness map
        # Appearance
        self.checkBox_3.setChecked(self.settings['map']['appearance']['include_stations'])
        self.comboBox_3.setCurrentIndex(self.comboBox_3.findText(self.settings['map']['appearance']['plotting_method']))
        self.comboBox_4.setCurrentIndex(self.comboBox_4.findText(self.settings['map']['appearance']['colormap']))
        self.lineEdit_10.setText(self.settings['map']['appearance']['station_marker'])
        self.lineEdit_9.setText(self.settings['map']['appearance']['station_marker_color'])
        # Shapefiles
        self.checkBox_4.setChecked(self.settings['map']['shapefiles']['include'])
        self.lineEdit_11.setText(self.settings['map']['shapefiles']['path'])

    def close(self):
        self.done(0)
        
    def save_settings(self):

        settings = {'ccp':{'appearance':{'include_stations':self.checkBox_6.isChecked(),
                                         'plotting_method':self.comboBox_6.currentText(),
                                         'colormap':self.comboBox_5.currentText(),
                                         'station_marker':self.lineEdit_14.text(),
                                         'station_marker_color':self.lineEdit_13.text()},
                           'shapefiles':{'include':self.checkBox_5.isChecked(),
                                         'path':self.lineEdit_12.text()},
                           'computation':{'earth_model':self.comboBox_7.currentText(),
                                          'stacking_method':self.comboBox_8.currentText()}},
                    'rfs':{'appearance':{'line_color':self.lineEdit.text(),
                                         'line_width':self.doubleSpinBox.value(),
                                         'positive_fill_color':self.lineEdit_3.text(),
                                         'negative_fill_color':self.lineEdit_2.text()},
                           'computation':{'normalize':self.checkBox.isChecked(),
                                          'w0':self.doubleSpinBox_2.value(),
                                          'time_shift':self.doubleSpinBox_3.value()},
                           'stacking':{'ref_slowness':self.doubleSpinBox_8.value()}},
                    'hk':{'appearance':{'plotting_method':self.comboBox.currentText(),
                                        'colormap':self.comboBox_2.currentText(),
                                        'line_color':self.lineEdit_7.text(),
                                        'ser_color':self.lineEdit_8.text()},
                          'computation':{'semblance_weighting':self.checkBox_2.isChecked(),
                                         'H_points':self.spinBox.value(),
                                         'k_points':self.spinBox_2.value(),
                                         'avg_vp':self.doubleSpinBox_5.value()},
                          'theoretical_atimes':{'ref_slowness':self.doubleSpinBox_6.value(),
                                                'avg_vp':self.doubleSpinBox_7.value()}},
                    'map':{'appearance':{'include_stations':self.checkBox_3.isChecked(),
                                         'plotting_method':self.comboBox_3.currentText(),
                                         'colormap':self.comboBox_4.currentText(),
                                         'station_marker':self.lineEdit_10.text(),
                                         'station_marker_color':self.lineEdit_9.text()},
                           'shapefiles':{'include':self.checkBox_4.isChecked(),
                                         'path':self.lineEdit_11.text()}}}
        
        pickle.dump(settings, open(CONFIG_PATH, 'wb'))
    
    def reset_to_defaults(self):
        qm = QtWidgets.QMessageBox
        ret = qm.question(self,'', "Are you sure to reset all the values?", qm.Yes | qm.No)
        if ret == qm.Yes:
            self.settings = mwu.read_preferences(return_defaults=True)
            self.read_settings()

class AboutDialog(QtWidgets.QDialog):
    def __init__(self):
        super(AboutDialog, self).__init__()
        uic.loadUi('rfun/ui/RfunDialogsAbout.ui', self)