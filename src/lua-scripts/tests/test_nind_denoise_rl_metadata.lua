-- Metadata validation tests for nind_denoise_rl.lua
local dt = require "darktable"
local du = require "lib/dtutils"
local df = require "lib/dtutils.file"
local nind = require "nind_denoise_rl"

local function make_temp_image()
  local tmp = os.tmpname()..".tif"
  os.execute("convert -size 100x100 xc:white "..tmp)
  return tmp
end

dt.test("metadata_export_success", function()
  local test_img = make_temp_image()
  local test_sidecar = test_img..".xmp"
  os.execute("echo '<xmp>test</xmp>' > "..test_sidecar)
  
  local success = pcall(function()
    dt.images.import(test_img)
    local image = dt.database[#dt.database]
    image:export_metadata(test_img.."_exported.tif", "XMP")
  end)
  
  dt.assert(success, "Metadata export failed")
  dt.assert(df.file_exists(test_img.."_exported.xmp"), 
    "XMP sidecar not created")
end)

dt.test("metadata_fallback", function()
  local test_img = make_temp_image()
  local image = dt.database[#dt.database]
  
  -- Force API failure
  local original_export = dt.Image.export_metadata
  dt.Image.export_metadata = function() error("simulated failure") end
  
  local success, err = pcall(function()
    nind.store(nil, image, {extension="tif"}, test_img, 1, 1, nil, {
      output_folder = "/tmp",
      denoise_enabled = false,
      rl_deblur_enabled = false
    })
  end)
  
  -- Restore original function
  dt.Image.export_metadata = original_export
  
  dt.assert(not success, "Fallback test should fail")
  dt.assert(string.find(err, "NDERR-1005"), 
    "Did not throw METADATA_FAILURE error")
end)

dt.test("windows_path_handling", function()
  if dt.configuration.running_os ~= "windows" then return end
  
  local win_path = "C:\\test\\path with spaces\\image.xmp"
  local sanitized = df.sanitize_filename(win_path)
  dt.assert(not string.find(sanitized, " "), 
    "Failed to sanitize Windows path")
  dt.assert(string.find(sanitized, "/"), 
    "Failed to convert path separators")
end)