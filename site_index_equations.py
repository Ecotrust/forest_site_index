"""
Implementation of several equations used to estimate site index, the height of
dominant trees at a particular age, generally in an even-aged stand.
"""

import numpy as np
import pandas as pd


def king1966_df(height, bh_age):
    """King, James E. 1966. Site index curves for Douglas-fir in the Pacific
    Northwest. Weyerhaeuser Forestry Paper No. 8. Centralia, WA. Weyerhaeuser
    Forestry Research Center. 49p.

    Parameters
    ----------
    height : numeric
        total tree height, in feet
    bh_age : numeric
        age of tree measured at breast height, in years

    Returns
    -------
    site_index : numeric
        estimated site index (estimated tree height, in feet, at age 50)
    """
    site_index = 4.5 + 2500/((bh_age**2)/((height - 4.5) + 0.954038 - \
                 0.0558178*bh_age + 0.000733819*bh_age**2)/(0.109757 + \
                 0.00792236*bh_age + 0.000197693*bh_age**2))

    return site_index


def cochran1979_gf(height, bh_age):
    """Cochran, P.H. 1979. Site index and height growth curves for managed,
    even-aged stands of white or grand fir east of the Cascades in Oregon and
    Washington. Res. Pap. PNW-252. Portland, OR: Forest Service, Pacific
    Northwest Forest and Range Experiment Station. 13 p.

    Parameters
    ----------
    height : numeric
        total tree height, in feet
    bh_age : numeric
        age of tree measured at breast height, in years

    Returns
    -------
    site_index : numeric
        estimated site index (estimated tree height, in feet, at age 50)
    """
    x1 = 3.8886 - 1.8017*np.log(bh_age) + 0.2105*(np.log(bh_age)**2) - \
         0.0000002885*(np.log(bh_age)**9) + 0.000000000000000001187 * \
         (np.log(bh_age)**24)

    x2 = -0.30935 + 1.2383*np.log(bh_age) + 0.001762*(np.log(bh_age)**4) - \
         0.0000054*(np.log(bh_age)**9) + 0.0000002046*(np.log(bh_age)**11) - \
         0.000000000000404*(np.log(bh_age)**18)

    site_index = (height - 4.5) * np.exp(x1) - np.exp(x1) * np.exp(x2) + 89.43

    return site_index


def wiley1978_wh(height, bh_age):
    """
    Wiley, Kenneth N. 1978. Site index tables for western hemlock in the
    Pacific Northwest. For. Pap. No. 17. Centralia, WA: Weyerhaeuser Forestry
    Research Center. 28 p.

    Parameters
    ----------
    height : numeric
        total tree height, in feet
    bh_age : numeric
        age of tree measured at breast height, in years

    Returns
    -------
    site_index : numeric
        estimated site index (estimated tree height, in feet, at age 50)
    """
    site_index = 2500 * ((height - 4.5) * (0.1394 + 0.0137 * bh_age + 0.00007 \
                 * bh_age**2) / (bh_age**2 - (height - 4.5) * (-1.7307 - \
                 0.0616 * bh_age + 0.00192 * bh_age**2)))

    return site_index


def harrington1986_ra(height, bh_age):
    """
    Harrington, Constance A.; Curtis, Robert O. 1986. Height growth and site
    index curves for red alder. Res. Pap. PNW-358. Portland, OR: Forest
    Service, Pacific Northwest Forest and Range Experiment Station. 14 p.

    Parameters
    ----------
    height : numeric
        total tree height, in feet
    bh_age : numeric
        age of tree measured at breast height, in years

    Returns
    -------
    site_index : numeric
        estimated site index (estimated tree height, in feet, at age 20)
    """
    # authors recommended adding 1 year to breast-height age to get total age
    age = bh_age + 1
    a = 54.1850 - 4.61694 * age + 0.11065 * age**2 - 0.0007633 * age**3
    b = 1.25934 - 0.012989 * age + 3.5220 * (1/age)**3

    site_index = a + b * height

    return site_index


def farr1984_ss(height, bh_age):
    """
    Farr, Wilbur A. 1984. Site index and height growth curves for unmanaged
    even-aged stands of western hemlock and Sitka spruce in southeast Alaska.
    Res. Pap. PNW-326. Portland, OR: Forest Service, Pacific Northwest Forest
    and Range Experiment Station. 26 p.

    Parameters
    ----------
    height : numeric
        total tree height, in feet
    bh_age : numeric
        age of tree measured at breast height, in years

    Returns
    -------
    site_index : numeric
        estimated site index (estimated tree height, in feet, at age 50)
    """
    # Farr used a table of coefficients that need to be looked up by age
    AGE_LOOKUP = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                  25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                  40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
                  55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
                  70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84,
                  85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
                  100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
                  112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123,
                  124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135,
                  136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147,
                  148, 149, 150]

    A2_LOOKUP = [43.22, 42.069, 40.753, 39.327, 37.827, 36.279, 34.705, 33.12,
                 31.536, 29.963, 28.408, 26.879, 25.38, 23.914, 22.486, 21.098,
                 19.751, 18.448, 17.189, 15.975, 14.806, 13.682, 12.603,
                 11.569, 10.58, 9.634, 8.731, 7.87, 7.05, 6.271, 5.53, 4.826,
                 4.16, 3.529, 2.932, 2.368, 1.836, 1.334, 0.862, 0.418, 0,
                 -0.391, -0.758, -1.102, -1.423, -1.723, -2.003, -2.264,
                 -2.507, -2.732, -2.942, -3.136, -3.316, -3.482, -3.635,
                 -3.777, -3.908, -4.028, -4.138, -4.239, -4.332, -4.418,
                 -4.496, -4.567, -4.633, -4.692, -4.747, -4.798, -4.844,
                 -4.886, -4.925, -4.961, -4.995, -5.026, -5.056, -5.083,
                 -5.11, -5.135, -5.159, -5.182, -5.205, -5.228, -5.25, -5.273,
                 -5.295, -5.318, -5.341, -5.364, -5.388, -5.412, -5.437,
                 -5.463, -5.489, -5.515, -5.543, -5.57, -5.598, -5.627,
                 -5.656, -5.685, -5.714, -5.743, -5.772, -5.801, -5.83,
                 -5.858, -5.885, -5.912, -5.937, -5.961, -5.984, -6.005,
                 -6.024, -6.041, -6.055, -6.067, -6.075, -6.081, -6.082,
                 -6.08, -6.073, -6.062, -6.046, -6.024, -5.997, -5.964,
                 -5.925, -5.878, -5.825, -5.764, -5.694, -5.617, -5.53,
                 -5.435, -5.329, -5.213, -5.087, -4.95, -4.801, -4.64, -4.467]

    B2_LOOKUP = [2.33, 2.143, 1.998, 1.884, 1.792, 1.716, 1.652, 1.598, 1.551,
                 1.51, 1.474, 1.441, 1.412, 1.385, 1.36, 1.338, 1.316, 1.296,
                 1.278, 1.26, 1.243, 1.227, 1.211, 1.197, 1.182, 1.169, 1.155,
                 1.142, 1.13, 1.117, 1.105, 1.094, 1.082, 1.071, 1.06, 1.05,
                 1.039, 1.029, 1.019, 1.01, 1, 0.991, 0.982, 0.973, 0.964,
                 0.955, 0.947, 0.939, 0.93, 0.923, 0.916, 0.907, 0.9, 0.893,
                 0.885, 0.878, 0.872, 0.865, 0.858, 0.852, 0.846, 0.84, 0.834,
                 0.828, 0.822, 0.817, 0.811, 0.806, 0.801, 0.796, 0.791,
                 0.786, 0.781, 0.777, 0.772, 0.768, 0.764, 0.76, 0.756, 0.752,
                 0.748, 0.744, 0.741, 0.737, 0.734, 0.73, 0.727, 0.724, 0.721,
                 0.718, 0.715, 0.712, 0.709, 0.707, 0.704, 0.702, 0.699,
                 0.697, 0.694, 0.692, 0.69, 0.688, 0.686, 0.684, 0.682, 0.68,
                 0.678, 0.676, 0.674, 0.672, 0.671, 0.669, 0.667, 0.665,
                 0.664, 0.662, 0.661, 0.659, 0.657, 0.656, 0.654, 0.653,
                 0.651, 0.65, 0.648, 0.647, 0.645, 0.643, 0.642, 0.64, 0.639,
                 0.637, 0.635, 0.634, 0.632, 0.63, 0.628, 0.626, 0.624, 0.622,
                 0.62]

    coefs = pd.DataFrame(data = {'age': AGE_LOOKUP,
                                 'a2': A2_LOOKUP,
                                 'b2': B2_LOOKUP}
                        ).set_index('age')
    # lookup the coefficients for user-provided breast-height age
    a2, b2 = coefs.loc[bh_age].values

    site_index = 4.5 + a2 + b2 * (height - 4.5)

    return site_index


def curtis1974_df(height, bh_age):
    """
    Curtis, Robert O.; Herman, Francis R.; DeMars, Donald J. 1974. Height
    growth and site index for Douglas-fir in high-elevation forests of the
    Oregon-Washington Cascades. Forest Science 20(4):307-316.

    Parameters
    ----------
    height : numeric
        total tree height, in feet
    bh_age : numeric
        age of tree measured at breast height, in years

    Returns
    -------
    site_index : numeric
        estimated site index (estimated tree height, in feet, at age 100)
    """
    a = np.where(bh_age <= 100,
                 0.010006 * (100 - bh_age)**2,  # when age <= 100
                 7.66772 * np.exp(0.95 * (100 / bh_age - 100)**2 )  # age > 100
                 )

    b = np.where(bh_age <= 100,
                 1.0 + 0.00549779 * (100 - bh_age) + 1.46842E-14 * \
                 (100 - bh_age)**7,  # when age <= 100
                 1.0 - 0.730948 * (np.log10(bh_age) - 2.0)**0.80  # age > 100
                 )

    site_index = a + b * (height - 4.5) + 4.5

    return site_index
