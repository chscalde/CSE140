# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 18:35:30 2020

@author: Cagan Bakirci
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

df = pd.read_csv ("Data/train_final.csv", nrows = 100000)

#features = df.values
#values = features.max(0)
#features.max(0)


#DROP SOME COLUMNS#######################################################
#Get index and drop index
index_loc = df.columns.get_loc("index")
index = df.iloc[:,index_loc].values

#Get y and drop y
y_loc = df.columns.get_loc("HasDetections")
y = df.iloc[:,y_loc].values

#Get MachineIdentifier and drop MachineIdentifier
id_loc = df.columns.get_loc("MachineIdentifier")
ID = df.iloc[:,id_loc].values

#Drop these colums
df = df.drop(columns = ["MachineIdentifier", "HasDetections", "index"])
########################################################################




#NORMALIZE COLUMNS INDIVIDUALLY##########################################
#Normalize AppVersion
appVersion_loc = df.columns.get_loc("AppVersion")
appVersion = df.iloc[:,appVersion_loc].values
sc_appVersion = StandardScaler()
df.iloc[:,appVersion_loc] = sc_appVersion.fit_transform(appVersion.reshape(-1, 1))


#Normalize AVProductStatesIdentifier
AVPID_loc = df.columns.get_loc("AVProductStatesIdentifier")
AVP = df.iloc[:,AVPID_loc].values
sc_AVP = StandardScaler()
df.iloc[:,AVPID_loc] = sc_AVP.fit_transform(AVP.reshape(-1, 1))


#Normalize AvSigVersion
avSig_loc = df.columns.get_loc("AvSigVersion")
avSig = df.iloc[:,avSig_loc].values
sc_avSig = StandardScaler()
df.iloc[:,avSig_loc] = sc_avSig.fit_transform(avSig.reshape(-1, 1))


#Normalize Census_ChassisTypeName
chassisType_loc = df.columns.get_loc("Census_ChassisTypeName")
chassisType = df.iloc[:,chassisType_loc].values
sc_chassisType = StandardScaler()
df.iloc[:,chassisType_loc] = sc_chassisType.fit_transform(chassisType.reshape(-1, 1))


#Normalize Census_FirmwareManufacturerIdentifier
frameID_loc = df.columns.get_loc("Census_FirmwareManufacturerIdentifier")
frameID = df.iloc[:,frameID_loc].values
sc_frameID = StandardScaler()
df.iloc[:,frameID_loc] = sc_frameID.fit_transform(frameID.reshape(-1, 1))


#Normalize Census_FirmwareVersionIdentifier
frameVerID_loc = df.columns.get_loc("Census_FirmwareVersionIdentifier")
frameVerID = df.iloc[:,frameVerID_loc].values
sc_frameVerID = StandardScaler()
df.iloc[:,frameVerID_loc] = sc_frameVerID.fit_transform(frameVerID.reshape(-1, 1))

###

#Normalize Census_InternalPrimaryDiagonalDisplaySizeInInches
DisplaySize_loc = df.columns.get_loc("Census_InternalPrimaryDiagonalDisplaySizeInInches")
DisplaySize = df.iloc[:,DisplaySize_loc].values
sc_DisplaySize = StandardScaler()
df.iloc[:,DisplaySize_loc] = sc_DisplaySize.fit_transform(DisplaySize.reshape(-1, 1))


#Normalize Census_InternalPrimaryDisplayResolutionHorizontal
drHorizontal_loc = df.columns.get_loc("Census_InternalPrimaryDisplayResolutionHorizontal")
drHorizontal = df.iloc[:,drHorizontal_loc].values
sc_drHorizontal = StandardScaler()
df.iloc[:,drHorizontal_loc] = sc_drHorizontal.fit_transform(drHorizontal.reshape(-1, 1))


#Normalize Census_InternalPrimaryDisplayResolutionVertical
drVertical_loc = df.columns.get_loc("Census_InternalPrimaryDisplayResolutionVertical")
drVertical = df.iloc[:,drVertical_loc].values
sc_drVertical = StandardScaler()
df.iloc[:,drVertical_loc] = sc_drVertical.fit_transform(drVertical.reshape(-1, 1))


#Normalize Census_MDC2FormFactor
md2_loc = df.columns.get_loc("Census_MDC2FormFactor")
md2 = df.iloc[:,md2_loc].values
sc_md2 = StandardScaler()
df.iloc[:,md2_loc] = sc_md2.fit_transform(md2.reshape(-1, 1))


#Normalize Census_OEMModelIdentifier
oem_loc = df.columns.get_loc("Census_OEMModelIdentifier")
oem = df.iloc[:,oem_loc].values
sc_oem = StandardScaler()
df.iloc[:,oem_loc] = sc_oem.fit_transform(oem.reshape(-1, 1))


#Normalize Census_OEMNameIdentifier
oemName_loc = df.columns.get_loc("Census_OEMNameIdentifier")
oemName = df.iloc[:,oemName_loc].values
sc_oemName = StandardScaler()
df.iloc[:,oemName_loc] = sc_oemName.fit_transform(oemName.reshape(-1, 1))


#Normalize Census_OSBranch
os_loc = df.columns.get_loc("Census_OSBranch")
os = df.iloc[:,os_loc].values
sc_os = StandardScaler()
df.iloc[:,os_loc] = sc_os.fit_transform(os.reshape(-1, 1))


#Normalize Census_OSBuildNumber
osNo_loc = df.columns.get_loc("Census_OSBuildNumber")
osNo = df.iloc[:,osNo_loc].values
sc_osNo = StandardScaler()
df.iloc[:,osNo_loc] = sc_osNo.fit_transform(osNo.reshape(-1, 1))


#Normalize Census_OSBuildRevision
osRev_loc = df.columns.get_loc("Census_OSBuildRevision")
osRev = df.iloc[:,osRev_loc].values
sc_osRev = StandardScaler()
df.iloc[:,osRev_loc] = sc_osRev.fit_transform(osRev.reshape(-1, 1))


#Normalize Census_OSInstallLanguageIdentifier
osLan_loc = df.columns.get_loc("Census_OSInstallLanguageIdentifier")
osLan = df.iloc[:,osLan_loc].values
sc_osLan = StandardScaler()
df.iloc[:,osLan_loc] = sc_osLan.fit_transform(osLan.reshape(-1, 1))



#Normalize Census_OSUILocaleIdentifier
osui_loc = df.columns.get_loc("Census_OSUILocaleIdentifier")
osui = df.iloc[:,osui_loc].values
sc_osui = StandardScaler()
df.iloc[:,osui_loc] = sc_osui.fit_transform(osui.reshape(-1, 1))


#Normalize Census_OSVersion
osVer_loc = df.columns.get_loc("Census_OSVersion")
osVer = df.iloc[:,osVer_loc].values
sc_osVer = StandardScaler()
df.iloc[:,osVer_loc] = sc_osVer.fit_transform(osVer.reshape(-1, 1))


#Normalize Census_PrimaryDiskTotalCapacity
totalCap_loc = df.columns.get_loc("Census_PrimaryDiskTotalCapacity")
totalCap = df.iloc[:,totalCap_loc].values
sc_totalCap = StandardScaler()
df.iloc[:,totalCap_loc] = sc_totalCap.fit_transform(totalCap.reshape(-1, 1))

###


#Normalize Census_ProcessorModelIdentifier
pMID_loc = df.columns.get_loc("Census_ProcessorModelIdentifier")
pMID = df.iloc[:,pMID_loc].values
sc_pMID = StandardScaler()
df.iloc[:,pMID_loc] = sc_pMID.fit_transform(pMID.reshape(-1, 1))


#Normalize Census_SystemVolumeTotalCapacity
totalCap_loc = df.columns.get_loc("Census_SystemVolumeTotalCapacity")
totalCap = df.iloc[:,totalCap_loc].values
sc_totalCap = StandardScaler()
df.iloc[:,totalCap_loc] = sc_totalCap.fit_transform(totalCap.reshape(-1, 1))


#Normalize Census_TotalPhysicalRAM
totalRAM_loc = df.columns.get_loc("Census_TotalPhysicalRAM")
totalRAM = df.iloc[:,totalRAM_loc].values
sc_totalRAM = StandardScaler()
df.iloc[:,totalRAM_loc] = sc_totalRAM.fit_transform(totalRAM.reshape(-1, 1))


#Normalize CityIdentifier
cityID_loc = df.columns.get_loc("CityIdentifier")
cityID = df.iloc[:,cityID_loc].values
sc_cityID = StandardScaler()
df.iloc[:,cityID_loc] = sc_cityID.fit_transform(cityID.reshape(-1, 1))


#Normalize CountryIdentifier
cID_loc = df.columns.get_loc("CountryIdentifier")
cID = df.iloc[:,cID_loc].values
sc_cID = StandardScaler()
df.iloc[:,cID_loc] = sc_cID.fit_transform(cID.reshape(-1, 1))


#Normalize GeoNameIdentifier
geoName_loc = df.columns.get_loc("GeoNameIdentifier")
geoName = df.iloc[:,geoName_loc].values
sc_geoName = StandardScaler()
df.iloc[:,geoName_loc] = sc_geoName.fit_transform(geoName.reshape(-1, 1))


#Normalize IeVerIdentifier
ie_loc = df.columns.get_loc("IeVerIdentifier")
ie = df.iloc[:,ie_loc].values
sc_ie = StandardScaler()
df.iloc[:,ie_loc] = sc_ie.fit_transform(ie.reshape(-1, 1))


#Normalize LocaleEnglishNameIdentifier
engID_loc = df.columns.get_loc("LocaleEnglishNameIdentifier")
engID = df.iloc[:,engID_loc].values
sc_engID = StandardScaler()
df.iloc[:,engID_loc] = sc_engID.fit_transform(engID.reshape(-1, 1))


#Normalize OrganizationIdentifier
orgID_loc = df.columns.get_loc("OrganizationIdentifier")
orgID = df.iloc[:,orgID_loc].values
sc_orgID = StandardScaler()
df.iloc[:,orgID_loc] = sc_orgID.fit_transform(orgID.reshape(-1, 1))


#Normalize OsBuild
osBuild_loc = df.columns.get_loc("OsBuild")
osBuild = df.iloc[:,osBuild_loc].values
sc_osBuild = StandardScaler()
df.iloc[:,osBuild_loc] = sc_osBuild.fit_transform(osBuild.reshape(-1, 1))


#Normalize OsBuildLab
osBuildLab_loc = df.columns.get_loc("OsBuildLab")
osBuildLab = df.iloc[:,osBuildLab_loc].values
sc_osBuildLab = StandardScaler()
df.iloc[:,osBuildLab_loc] = sc_osBuildLab.fit_transform(osBuildLab.reshape(-1, 1))


#Normalize OsSuite
osSuite_loc = df.columns.get_loc("OsSuite")
osSuite = df.iloc[:,osSuite_loc].values
sc_osSuite = StandardScaler()
df.iloc[:,osSuite_loc] = sc_osSuite.fit_transform(osSuite.reshape(-1, 1))



#Normalize Wdft_RegionIdentifier
wdft_loc = df.columns.get_loc("Wdft_RegionIdentifier")
wdft = df.iloc[:,wdft_loc].values
sc_wdft = StandardScaler()
df.iloc[:,wdft_loc] = sc_wdft.fit_transform(wdft.reshape(-1, 1))
########################################################################




#Get the dfeature_importance files from the RandomForests
feature_imp_1 = pd.read_csv ("FeatureImportance1.csv")
feature_imp_25 = pd.read_csv("RandomForrest/25/FeatureImportance25.csv")
feature_imp_50 = pd.read_csv("RandomForrest/50/FeatureImportance50.csv")
feature_imp_70 = pd.read_csv("RandomForrest/70/FeatureImportance70.csv")


#Create a df with most important 36 features

df36 = df.drop(columns = ["Census_DeviceFamily", "Census_IsPortableOperatingSystem", "SMode", "HasTpm",
                          "Platform", "OsVer", "IsSxsPassiveMode", "Census_IsAlwaysOnAlwaysConnectedCapable",
                          "Census_IsVirtualDevice", "Census_IsPenCapable", "AVProductsEnabled",
                          "Census_OSArchitecture", "Processor", "Firewall", "IsProtected", "RtpStateBitfield",
                          "Census_FlightRing", "OsSuite", "Census_ProcessorManufacturerIdentifier",
                          "Census_OSSkuName", "Census_OSEdition", "Census_IsTouchEnabled", "SkuEdition",
                          "OsPlatformSubRelease", "OsBuild", "Census_HasOpticalDiskDrive", "Census_OSBuildNumber",
                          "Census_PowerPlatformRoleName", "Census_GenuineStateName", "Census_PrimaryDiskTypeName",
                          "Census_MDC2FormFactor", "Census_IsSecureBootEnabled", "Census_MDC2FormFactor_new"])

#Create a df with most important 16 features

df16 = df.drop(columns = ["Census_DeviceFamily", "Census_IsPortableOperatingSystem", "SMode", "HasTpm",
                          "Platform", "OsVer", "IsSxsPassiveMode", "Census_IsAlwaysOnAlwaysConnectedCapable",
                          "Census_IsVirtualDevice", "Census_IsPenCapable", "AVProductsEnabled",
                          "Census_OSArchitecture", "Processor", "Firewall", "IsProtected", "RtpStateBitfield",
                          "Census_FlightRing", "OsSuite", "Census_ProcessorManufacturerIdentifier",
                          "Census_OSSkuName", "Census_OSEdition", "Census_IsTouchEnabled", "SkuEdition",
                          "OsPlatformSubRelease", "OsBuild", "Census_HasOpticalDiskDrive", "Census_OSBuildNumber",
                          "Census_PowerPlatformRoleName", "Census_GenuineStateName", "Census_PrimaryDiskTypeName",
                          "Census_MDC2FormFactor", "Census_IsSecureBootEnabled", "Census_MDC2FormFactor_new", 
                          "Census_OSBranch", "Wdft_IsGamer", "Census_ActivationChannel", "Census_ProcessorCoreCount",
                          "Census_OSWUAutoUpdateOptionsName", "IeVerIdentifier", "Census_ChassisTypeName", "OsBuildLab",
                          "Census_InternalPrimaryDisplayResolutionVertical", "Census_InternalPrimaryDisplayResolutionHorizontal",
                          "AVProductsInstalled", "Census_TotalPhysicalRAM", "EngineVersion", "AppVersion", "Census_OSInstallLanguageIdentifier",
                          "Census_OSUILocaleIdentifier", "OrganizationIdentifier", "Census_FirmwareManufacturerIdentifier",
                          "Census_OSInstallTypeName", "Wdft_RegionIdentifier", ])

#Create a df with most important 4 features

df4 = df.drop(columns = ["Census_DeviceFamily", "Census_IsPortableOperatingSystem", "SMode", "HasTpm",
                          "Platform", "OsVer", "IsSxsPassiveMode", "Census_IsAlwaysOnAlwaysConnectedCapable",
                          "Census_IsVirtualDevice", "Census_IsPenCapable", "AVProductsEnabled",
                          "Census_OSArchitecture", "Processor", "Firewall", "IsProtected", "RtpStateBitfield",
                          "Census_FlightRing", "OsSuite", "Census_ProcessorManufacturerIdentifier",
                          "Census_OSSkuName", "Census_OSEdition", "Census_IsTouchEnabled", "SkuEdition",
                          "OsPlatformSubRelease", "OsBuild", "Census_HasOpticalDiskDrive", "Census_OSBuildNumber",
                          "Census_PowerPlatformRoleName", "Census_GenuineStateName", "Census_PrimaryDiskTypeName",
                          "Census_MDC2FormFactor", "Census_IsSecureBootEnabled", "Census_MDC2FormFactor_new", 
                          "Census_OSBranch", "Wdft_IsGamer", "Census_ActivationChannel", "Census_ProcessorCoreCount",
                          "Census_OSWUAutoUpdateOptionsName", "IeVerIdentifier", "Census_ChassisTypeName", "OsBuildLab",
                          "Census_InternalPrimaryDisplayResolutionVertical", "Census_InternalPrimaryDisplayResolutionHorizontal",
                          "AVProductsInstalled", "Census_TotalPhysicalRAM", "EngineVersion", "AppVersion", "Census_OSInstallLanguageIdentifier",
                          "Census_OSUILocaleIdentifier", "OrganizationIdentifier", "Census_FirmwareManufacturerIdentifier",
                          "Census_OSInstallTypeName", "Wdft_RegionIdentifier", "Census_PrimaryDiskTotalCapacity",
                          "Census_OEMNameIdentifier", "Census_OSBuildRevision", "AVProductStatesIdentifier",
                          "Census_OSVersion", "LocaleEnglishNameIdentifier", "GeoNameIdentifier", "Census_InternalPrimaryDiagonalDisplaySizeInInches",
                          "CountryIdentifier", "Census_ProcessorModelIdentifier", "Census_OEMModelIdentifier", "Census_FirmwareVersionIdentifier"])



