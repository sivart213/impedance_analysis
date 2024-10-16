# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 17:05:01 2018.

@author: JClenney

General function file
"""

# Standard library imports
import ctypes
import itertools
import os
import sys
from pathlib import Path


# %% Path resolving functions
def find_drives(exclude_nonlocal=True, exclude_hidden=True):
    """
    Finds and returns a list of drive paths available on the system.

    This function dynamically identifies available drives based on the operating system.
    For Windows, it includes both local and network drives. For Linux and Darwin (macOS),
    it searches for directories matching certain patterns and includes the home directory
    and root. The function can exclude non-local (network) drives and hidden drives based
    on the parameters provided.

    Parameters:
    - exclude_nonlocal (bool): If True, excludes network drives from the result. Default is True.
    - exclude_hidden (bool): If True, excludes drives that are marked as hidden. Default is True.

    Returns:
    - list: A list of pathlib.Path objects representing the paths to the drives found.

    Note:
    - On Windows, network drives are detected using the win32net module and local drives are
    identified using ctypes to call GetLogicalDrives. The function filters out non-local or
    hidden drives based on the parameters.
    - On Linux, it looks for directories under "/" that match "m*/*" and checks if they are
    directories with contents. It also adds the home directory and root ("/") to the list.
    - On Darwin (macOS), it performs a similar search under "/" for directories matching "Vol*/*".
    - For other platforms, it defaults to adding the home directory and its root.
    - The function also provides an option to exclude drives that are symbolic links or
    do not match their realpath, aiming to filter out non-local drives.
    - Hidden drives are determined by a simple check if the drive's name ends with ".hidden".
    """
    if sys.platform.startswith("win"):
        drives = detect_windows_drives(exclude_nonlocal=exclude_nonlocal)
    elif sys.platform.startswith("linu"):
        drives = detect_posix_drives("m*/*", exclude_nonlocal)
    elif sys.platform.startswith("darw"):
        drives = detect_posix_drives("Vol*/*", exclude_nonlocal)
    else:
        drives = [Path.home(), Path(Path.home().parts[0])]

    if exclude_hidden:
        drives = [dr for dr in drives if not str(dr).lower().endswith(".hidden")]

    return drives


def detect_windows_drives(letters="ABCDEFGHIJKLMNOPQRSTUVWXYZ", exclude_nonlocal=True):
    """
    Detects and returns a list of drive paths available on a Windows system, with options to exclude non-local drives.

    This function identifies available drives by querying the system for logical drives and network drives. It uses
    the win32net module to enumerate network drives and ctypes to access the GetLogicalDrives Windows API function,
    which provides a bitmask representing the drives available on the system. The function can be configured to
    exclude non-local (network) drives based on the parameters provided.

    Parameters:
    - letters (str, optional): A string containing the uppercase alphabet letters used to check for drive presence.
      Default is "ABCDEFGHIJKLMNOPQRSTUVWXYZ".
    - exclude_nonlocal (bool, optional): If True, excludes network drives from the result. Default is True.

    Returns:
    - list of pathlib.Path: A list of Path objects representing the paths to the drives found on the system.
      Each Path object corresponds to a drive root (e.g., C:/).

    Note:
    - Network drives are detected using the win32net.NetUseEnum function, which enumerates all network connections.
      The function checks each connection's status to determine if it should be considered a drive.
    - Local drives are identified by converting the bitmask returned by GetLogicalDrives into drive letters.
    - If exclude_nonlocal is True, the function filters out drives that are mapped to network locations.
    """
    import win32net

    resume = 0
    net_dr = []
    while 1:
        net_res, _, resume = win32net.NetUseEnum(None, 0, resume)
        for dr in net_res:
            net_dr.append(Path(dr["local"]))
            net_dr.append(Path(dr["remote"]))
        if not resume:
            break

    drive_bitmask = ctypes.cdll.kernel32.GetLogicalDrives()
    drives = list(
        map(
            Path,
            map(
                "{}:/".format,
                itertools.compress(
                    letters,
                    map(lambda x: ord(x) - ord("0"), bin(drive_bitmask)[:1:-1]),
                ),
            ),
        )
    )
    if exclude_nonlocal:
        drives = [dr for dr in drives if Path(dr.drive) not in net_dr]
        drives = [dr for dr in drives if os.path.realpath(dr) == str(dr)]
    return drives


def detect_posix_drives(pattern="m*/*", exclude_nonlocal=True):
    """
    Detects and returns a list of drive paths available on POSIX-compliant systems (e.g., Linux, macOS).

    This function identifies available drives by searching for directories that match a specified pattern
    at the root ("/") directory level. It then checks if these directories are actual mount points by
    verifying they contain subdirectories. Optionally, it can exclude drives that are mounted from network
    locations based on the realpath comparison, aiming to filter out non-local drives.

    Parameters:
    - pattern (str, optional): The glob pattern used to identify potential drives at the root directory.
      Default is "m*/*", which aims to target mnt and media directories having at least one subdirectory typical of Linux structures.  Alternatively, utilize "Vol*/*" for macOS.
    - exclude_nonlocal (bool, optional): If True, excludes drives that do not have their realpath matching
      their path, which typically indicates network-mounted drives. Default is True.

    Returns:
    - list of pathlib.Path: A list of Path objects representing the mount points of the drives found on the
      system. Each Path object corresponds to a drive's mount point.

    Note:
    - The function initially searches for directories at the root ("/") that match the specified pattern.
    - It then filters these directories to include only those that contain at least one subdirectory,
      under the assumption that a valid drive mount point will have subdirectories.
    - The home directory and the root directory are always included in the list of drives.
    - If exclude_nonlocal is True, the function filters out drives that are mounted from network locations
      by comparing each drive's realpath to its original path. Drives with differing realpaths are considered
      non-local and excluded from the results.
    """
    drives = [dr for dr in Path("/").glob(pattern) if dr.is_dir() and any(dr.iterdir())]
    for drn, dr in enumerate(drives):
        dr_f = [x for x in os.listdir(dr)]
        while len(dr_f) == 1:
            drives[drn] = dr / dr_f[0]
            dr_f = [x for x in drives[drn].iterdir()]
    drives.append(Path.home())
    drives.append(Path("/"))

    if exclude_nonlocal:
        drives = [dr for dr in drives if os.path.realpath(dr) == str(dr)]
    return drives
