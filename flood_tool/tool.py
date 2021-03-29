"""Locator functions to interact with geographic data"""
import pandas as pd
import numpy as np
from .geo import *


__all__ = ['Tool']

class Tool(object):
    """Class to interact with a postcode database file."""

    def __init__(self, postcode_file=None, risk_file=None, values_file=None):
        """
        Reads postcode and flood risk files and provides a postcode locator service.

        Parameters

        ---------
        postcode_file : str, optional
            Filename of a .csv file containing geographic location data for postcodes.
        risk_file : str, optional
            Filename of a .csv file containing flood risk data.
        postcode_file : str, optional
            Filename of a .csv file containing property value data for postcodes.
        """
        
        # read postcodes.csv
        if postcode_file == None:
            self.df_postcode_file = pd.read_csv("./resources/postcodes.csv")
        else:
            self.df_postcode_file = pd.read_csv(postcode_file)
        
        # formatting the postcode column of postcodes file
        self.df_postcode_file.Postcode = self.df_postcode_file.Postcode.str.replace(' ',
                                                                                    '')  # delete space in postcodes strings
        self.df_postcode_file.Postcode = self.df_postcode_file.Postcode.str.strip()
        self.df_postcode_file.Postcode = self.df_postcode_file.Postcode.str.upper()

        # read flood_probability.csv
        if risk_file == None:
            self.df_risk_file = pd.read_csv("./resources/flood_probability.csv")
        else:
            self.df_risk_file = pd.read_csv(risk_file)
        del self.df_risk_file['Unnamed: 0']

        # read property_value.csv
        if values_file == None:
            self.df_values_file = pd.read_csv("./resources/property_value.csv")
        else:
            self.df_values_file = pd.read_csv(values_file)

        # formatting the postcode column of property file
        self.df_values_file.Postcode = self.df_values_file.Postcode.str.replace(' ',
                                                                                '')  # delete space in postcodes strings
        self.df_values_file.Postcode = self.df_values_file.Postcode.str.strip()
        self.df_values_file.Postcode = self.df_values_file.Postcode.str.upper()

        # merge two - ensures entering invalid code returns nan but entering code with no property value returns 0
        self.cat_pst_values = self.df_postcode_file.set_index('Postcode').join(self.df_values_file.set_index('Postcode'))
        del self.cat_pst_values['Lat']
        del self.cat_pst_values['Long']
        self.cat_pst_values = self.cat_pst_values.fillna(0)


    def get_lat_long(self, postcodes):
        """Get an array of WGS84 (latitude, longitude) pairs from a list of postcodes.

        Parameters
        ----------

        postcodes: sequence of strs
            Ordered sequence of N postcode strings

        Returns
        -------

        ndarray
            Array of Nx2 (latitude, longitdue) pairs for the input postcodes.
            Invalid postcodes return [`numpy.nan`, `numpy.nan`].
        """
        # Clean the postcodes
        postcodes = [postcode.replace(' ', '').upper().strip() for postcode in postcodes]
        # Select columns to return
        indices = self.cat_pst_values.loc[postcodes, ['Latitude', 'Longitude']]
        return indices.values


    def get_easting_northing_flood_probability(self, easting, northing):
        """Get an array of flood risk probabilities from arrays of eastings and northings.

        Flood risk data is extracted from the Tool flood risk file. Locations
        not in a risk band circle return `Zero`, otherwise returns the name of the
        highest band it sits in.

        Parameters
        ----------

        easting: numpy.ndarray of floats
            OS Eastings of locations of interest
        northing: numpy.ndarray of floats
            Ordered sequence of postcodes

        Returns
        -------

        numpy.ndarray of strs
            numpy array of flood probability bands corresponding to input locations.
        """
        # Use the provided probabilities band file
        flood_prob = self.df_risk_file

        # Sort by order of probability from high to zero
        flood_prob['prob_4band'] = pd.Categorical(flood_prob['prob_4band'], ["High", "Medium", "Low",
                                                             "Very Low", "Zero"])
        flood_prob = flood_prob.sort_values(by=['prob_4band'])

        # Split columns into different variables
        cor1x = np.array(flood_prob.loc[:,'X'].values)
        cor1y = np.array(flood_prob.loc[:,'Y'].values)
        pro = np.array(flood_prob.loc[:,'prob_4band'].values)
        r = np.array(flood_prob.loc[:,'radius'].values)
        r = r*r
        
        # Initialize list to hold outputs
        outputpro = []
        for m, n in zip(easting, northing):
            # Compare distance squared  to radius squared to save on computing time
            r_relative = (cor1x - m) ** 2 + (cor1y - n) ** 2
            judge = r_relative < r
            
            if (len(pro[judge]) == 0):
                outputpro.append("Zero")
            else: # Since the file is sorted, the first hit is the greatest band
                outputpro.append(pro[judge][0])
         
        return np.array(outputpro)

    def get_sorted_flood_probability(self, postcodes):
        """Get an array of flood risk probabilities from a sequence of postcodes.

        Probability is ordered High>Medium>Low>Very low>Zero.
        Flood risk data is extracted from the `Tool` flood risk file.

        Parameters
        ----------

        postcodes: sequence of strs
            Ordered sequence of postcodes

        Returns
        -------

        pandas.DataFrame
            Dataframe of flood probabilities indexed by postcode and ordered from `High` to `Zero`,
            then by lexagraphic (dictionary) order on postcode. The index is named `Postcode`, the
            data column is named `Probability Band`. Invalid postcodes and duplicates
            are removed.
        """
        # Since we are returning postcodes here, we need to make them fit the requested format
        postcodes = [pc.upper() for pc in postcodes]
        postcodes = [pc.replace(' ', '') if len(pc) > 7 else pc for pc in postcodes]
        postcodes = [pc[0:2] + ' ' + pc[2:] if len(pc) == 5 else pc for pc in postcodes]
        postcodes = [pc[0:3] + ' ' + pc[3:] if (len(pc) == 6) and (pc[2] != ' ') else pc for pc in postcodes]

        # Postcodes -> Latitudes,Longitude -> Eastings,Northings -> Probabilities -> Sorted probabilities  
        lat_longs = self.get_lat_long(postcodes)
        east_norths = get_easting_northing_from_lat_long(lat_longs[:, 0], lat_longs[:, 1])
        probs = self.get_easting_northing_flood_probability(east_norths[0], east_norths[1])
        probs_sort = pd.DataFrame({'Postcode': postcodes, 'Probability Band': probs})
        probs_sort.set_index('Postcode', inplace=True)
        probs_sort['Probability Band'] = pd.Categorical(probs_sort['Probability Band'],
                                                       ["High", "Medium", "Low", "Very Low", "Zero"])
        probs_sort = probs_sort.sort_values(by=['Probability Band', 'Postcode'])
        
        # Drop duplicates and nans
        probs_sort = probs_sort[~probs_sort.index.duplicated(keep='first')].dropna()
        return probs_sort


    def get_flood_cost(self, postcodes):
        """Get an array of estimated cost of a flood event from a sequence of postcodes.
        Parameters
        ----------

        postcodes: sequence of strs
            Ordered collection of postcodes
        probability_bands: sequence of strs
            Ordered collection of flood probability bands

        Returns
        -------

        numpy.ndarray of floats
            array of floats for the pound sterling cost for the input postcodes.
            Invalid postcodes return `numpy.nan`.
        """
        # Clean up postcode input
        postcodes = [postcode.replace(' ', '').upper().strip() for postcode in postcodes]
        # Select relevant column, return numpy array
        indices = self.cat_pst_values.loc[postcodes, 'Total Value']
        return np.array(indices.values)


    def get_annual_flood_risk(self, postcodes, probability_bands):
        """Get an array of estimated annual flood risk in pounds sterling per year of a flood
        event from a sequence of postcodes and flood probabilities.

        Parameters
        ----------

        postcodes: sequence of strs
            Ordered collection of postcodes
        probability_bands: sequence of strs
            Ordered collection of flood probabilities

        Returns
        -------

        numpy.ndarray
            array of floats for the annual flood risk in pounds sterling for the input postcodes.
            Invalid postcodes return `numpy.nan`.
        """
        
        flood_risk = []
        reduce_cost = 0.05 #20%
        postcodes = [postcode.replace(' ', '').upper().strip() for postcode in postcodes]
        pro = np.array(probability_bands)
        pro[pro == "High"] = 1 / 10
        pro[pro == "Medium"] = 1 / 50
        pro[pro == "Low"] = 1 / 100
        pro[pro == "Very Low"] = 1 / 1000
        pro[pro == "Zero"] = 0.0
        flood_risk = self.cat_pst_values.loc[postcodes, 'Total Value'].values * pro.astype(float) * reduce_cost
        return np.array(flood_risk)


    def get_sorted_annual_flood_risk(self, postcodes):
        """Get a sorted pandas DataFrame of flood risks.

        Parameters
        ----------

        postcodes: sequence of strs
            Ordered sequence of postcodes

        Returns
        -------

        pandas.DataFrame
            Dataframe of flood risks indexed by (normalized) postcode and ordered by risk,
            then by lexagraphic (dictionary) order on the postcode. The index is named
            `Postcode` and the data column `Flood Risk`.
            Invalid postcodes and duplicates are removed.
        """
        # Let the previous function handle getting the probabilities
        flood_risk = self.get_sorted_flood_probability(postcodes)
        # Get the new postcodes that fit the style requirements from the previous function
        postcodes_new = flood_risk.index.values
        # Get risks, then sort them in place
        risk = self.get_annual_flood_risk(postcodes_new, flood_risk.values.reshape((-1)))
        probs_sort = pd.DataFrame({'Postcode': postcodes_new, 'Flood Risk': risk})
        annual_flood_risk = probs_sort.sort_values(by=['Flood Risk', 'Postcode'], ascending=[False, True])
        annual_flood_risk.set_index('Postcode', inplace=True)
        # Drop duplicates and nans
        annual_flood_risk = annual_flood_risk[~annual_flood_risk.index.duplicated(keep='first')].dropna()
        return annual_flood_risk