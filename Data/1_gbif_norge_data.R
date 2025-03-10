{
  #  GIBIF NORWAY-----
  #Author:Camila Pacheco
  #inc data:25/05/2021
  #Libraries ----
  
  library(tidyverse)
  library(data.table)
  library(rgbif)
  library(sf)
  library(raster)
  library(CoordinateCleaner)
  library(scrubr)
  library(countrycode)
  library(lubridate)
  library(bigreadr)
  library(geodata)
  
  #load("env/full_data.RData")
  
  #Configurations  ----
  tempfile(tmpdir = "D:\\Temp")
  
}


GetGBIFData <- function(data.request) {
  key <- occ_download_meta(data.request)$key
  filepath <- paste0(key, ".zip")
  if (file.exists(filepath)) {
    unzip(filepath)
    n.recs <- bigreadr::nlines(paste0(key, ".csv"))
    print(paste("occurrence file has", n.recs, "records"))
    occs <- bigreadr::big_fread1(
      paste0(key, ".csv"),
      every_nlines = 1000000,
      integer64 = "double",
      .transform = function(x) {
        res <-
          dplyr::select(
            x,
            c(
              "gbifID",
              "taxonRank",
              "family",
              "species",
              "decimalLongitude" ,
              "decimalLatitude",
              "elevation",
              "eventDate",
              "countryCode",
              "institutionCode",
              "publisher",
              "locality",
              "recordedBy",
              "coordinateUncertaintyInMeters",
              "elevationAccuracy"
            )
          )
        res <-
          dplyr::distinct(
            res,
            species,
            decimalLongitude,
            decimalLatitude,
            elevation,
            year,
            countryCode,
            elevationAccuracy ,
            .keep_all = T
          )
        return(res)
      }
    )
    file.remove(filepath)
    return(occs)
  } else {
    print(paste0("File ", filepath, " does not exist in working directory"))
    return(NULL)
  }
}

DownloadGBIF <- function(key, user, user.email, pwd, custom.shp) {
  if (is.null(custom.shp)) {
    print(paste("Must provide either a valid  SpatialPolygons* object"))
    stop()
  }
  else {
    aoi.wkt <- rgbif::gbif_bbox2wkt(bbox = st_bbox(custom.shp))
    data.request <- rgbif::occ_download(
      #Vascular plants
      pred_in("taxonKey", key),
      #only records with coordinates
      pred('hasCoordinate', TRUE),
      #format csv
      format = "SIMPLE_CSV",
      #shapefile limit
      pred_within(aoi.wkt),
      user = user,
      pwd = pwd,
      email = user.email
    )
    
  }
  #Sleep until query returns total number of records or it's status changes to KILLED
  current.status <- rgbif::occ_download_meta(data.request)$status
  
  #Check every if data is ready
  while (current.status != "SUCCEEDED" & current.status != "KILLED") {
    s <- system.time(Sys.sleep(50))
    current.status <- rgbif::occ_download_meta(data.request)$status
    print(paste("Download status is:", current.status))
    
  }
  
  data.grab <- rgbif::occ_download_get(data.request, overwrite = T)
  citation <- rgbif::gbif_citation(data.grab)$download
  
  occ.table <- GetGBIFData(data.request)
  return(list(occ.table = occ.table, citation = citation))
}

CleenGBIFDATA <- function(occ.list) {
  if (is.list(occ.list) == FALSE) {
    print(paste("Must provide either a valid  list object"))
    stop()
    
  }
  else{
    occ.table  <-  occ.list$occ.table %>%
      #Missing Data Values
      drop_na(species) %>%
      filter(!species == "") %>%
      #Incorrect coord Values
      coord_impossible() %>%
      #incomplete
      coord_incomplete() %>%
      #unlikely
      coord_unlikely() %>%
      #record date standardization
      drop_na(eventDate) %>%
      filter(!eventDate == "") %>%
      mutate(eventDate = as_date(eventDate)) %>%
      mutate(year = year(eventDate))
    
    #remove records with Geo problems
    
    occ.table.clean  <-
      clean_coordinates(
        x = occ.table %>% as_tibble() ,
        lon = "decimalLongitude",
        lat = "decimalLatitude",
        species = "species",
        countries = "countryCode",
        tests = c(
          "capitals",
          "centroids",
          "equal",
          "gbif",
          "institutions",
          "seas",
          "zeros"
        )
      ) %>%
      #Exclude problematic records
      as_tibble() %>%
      filter(!.summary == FALSE) %>%
      select(-starts_with("."))
    return(list(occ.table = occ.table.clean, citation = occ.list$citation))
  }
}

strip.edges <- function(x, rm.periods = FALSE) {
  if (!rm.periods) {
    x <- gsub("^[\\W\\d_]+|(?!\\.)[\\W\\d_]+$", "", x, perl = TRUE)
  } else {
    x <- gsub("^[\\W\\d_]+|[\\W\\d_]+$", "", x, perl = TRUE)
  }
  return(x)
}


# · GBIF keys ----

{
  gbif_user <- rstudioapi::askForPassword("my gbif username")
  gbif_email <- rstudioapi::askForPassword("my registred gbif e-mail")
  gbif_pwd <- rstudioapi::askForPassword("my gbif password")
}
# Download data ----
#library(geodata)
norway_shp <- gadm(country='NOR', level=0,path = tempdir())
plot(norway_shp)
norge <- st_as_sf(norway_shp)

{
  #. Key vascular plants----
  
  #shape mountains
  #norge <-
  #  st_read(
  #    "D:/GIS/Norway/gadm36_NOR_shp/gadm36_NOR_0.shp"
  #  )
  
  
  key <- name_backbone(name = "Tracheophyta", phylum = "Tracheophyte") %>%
    pull("phylumKey")
  
  occ_table <- DownloadGBIF(key,gbif_user, gbif_email,gbif_pwd,norge)
  
}

data_file = '0002901-250221090833381.csv'
# . Cleaning data ----

occs <- bigreadr::big_fread1(
  data_file, #file obtain from the download process insert here if you decided to manually download it
  every_nlines = 1000000, #1000000,
  integer64 = "double",
  .transform = function(x) {
    print(colnames(x))
    res <-
      dplyr::select(
        x,
        c(
          "gbifID",
          "taxonRank",
          "family",
          "species",
          "verbatimScientificNameAuthorship",
          "decimalLongitude" ,
          "decimalLatitude",
          "coordinateUncertaintyInMeters",
          "elevation",
          "elevationAccuracy",
          "year",
          "countryCode",
          "locality",
          "basisOfRecord",
          "institutionCode",
          "recordedBy",
          "coordinateUncertaintyInMeters",
          "elevationAccuracy",
          "issue"
        )
        
      )
    #print(colnames(occs))
    res <- res %>% dplyr::filter(year >= 1991 & year <= 2020)
    res <-dplyr::distinct(
        res,
        species,
        decimalLongitude,
        decimalLatitude,
        elevation,
        year,
        countryCode,
        elevationAccuracy ,
        .keep_all = T
      )
    return(res)
  }
)

occ.table.clean  <-  occs

occ.table.clean <- occ.table.clean %>%  drop_na(species) %>%
  #filtering species level
  dplyr::filter(!species == "") %>%
  #Incorrect coord Values
  coord_impossible() %>%
  #incomplete
  coord_incomplete() %>%
  #unlikely
  coord_unlikely() %>%
  #record date standardization
  drop_na(year) %>%
  filter(countryCode == "NO")


occ.table.clean  <-
  clean_coordinates(
    x = occ.table.clean %>% as_tibble() ,
    lon = "decimalLongitude",
    lat = "decimalLatitude",
    species = "species",
    countries = "countryCode",
    tests = c(
      "capitals",
      "centroids",
      "equal",
      "gbif",
      "institutions",
      "seas",
      "zeros"
    )
  ) %>%
  #Exclude problematic records
  as_tibble() %>%
  filter(!.summary == FALSE) %>%
  dplyr::select(-starts_with("."))


occ.table.clean <- occ.table.clean %>% st_as_sf(coords = c("decimalLongitude", "decimalLatitude"),
                                                crs = "+proj=longlat +datum=WGS84 +ellps=WGS84 +towgs84=0,0,0",
                                                remove = FALSE)
#norway shapefile

# Codigo modificado
#library(geodata)

norway_shp <- gadm(country = "NOR", level = 0, path = tempdir())
plot(norway_shp)  # Quick plot to check the map
# ---------------------------------------------------------------
#Norway <- st_read("C:/Users/lri042/Desktop/Bergen-PhD/GIS info/Norge_Ad_limit.shp")
Norway <- st_read("gadm41_NOR_shp/gadm41_NOR_0.shp")

Norway <-st_transform(Norway,crs = "+proj=longlat +datum=WGS84 +ellps=WGS84 +towgs84=0,0,0")
sf::sf_use_s2(FALSE)

occ.table.clean <- st_intersection(Norway,occ.table.clean)
occ.table.clean <- occ.table.clean %>% dplyr::select(-OBJECTID,-objtype,-navn,-oppdaterin,-kommunenum,-SHAPE_Leng,-SHAPE_Area)

#Norway DEM 1km res

dem <- raster::raster("C:/Users/lri042/Desktop/Bergen-PhD/GIS info/Norge/Norway_DEM/DEM_Nor_1km_f.tif")


occ.table.clean <- occ.table.clean %>%
  mutate(def_elevation = if_else(is.na(elevation),raster::extract(dem, .),elevation)) %>%
  st_drop_geometry() %>% as_tibble()


save.image("env/full_data.RData")
write.csv(occ.table.clean, "occ_table_clean.csv", sep = '\t', row.names = FALSE)


write.table(occ.table.clean, "occ_table_clean.tsv", sep = "\t", row.names = FALSE, quote = TRUE)
