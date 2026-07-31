; Inno Setup Script for FlowFI
; Configured for PyInstaller --onedir -> Inno Setup -> Windows MSIX Packaging Tool workflow

[Setup]
AppName=FlowFI
AppVersion=1.6.0
AppPublisher=FlowFI Team
AppPublisherURL=https://github.com/jameswilsenach/FlowFI
AppSupportURL=https://github.com/jameswilsenach/FlowFI
AppUpdatesURL=https://github.com/jameswilsenach/FlowFI

; Destination folder and output
DefaultDirName={autopf}\FlowFI
DefaultGroupName=FlowFI
OutputDir=.\Output
OutputBaseFilename=FlowFI_Setup
SetupIconFile=logo.ico
Compression=lzma2/max
SolidCompression=yes

; Supports per-user and per-machine installation (MSIX Packaging Tool friendly)
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; Point to the output folder of your PyInstaller --onedir build (dist\FlowFI)
Source: "logo.ico"; DestDir: "{app}"; Flags: ignoreversion
Source: "dist\FlowFI\FlowFI.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "dist\FlowFI\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\FlowFI"; Filename: "{app}\FlowFI.exe"; IconFilename: "{app}\logo.ico"
Name: "{group}\{cm:UninstallProgram,FlowFI}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\FlowFI"; Filename: "{app}\FlowFI.exe"; IconFilename: "{app}\logo.ico"; Tasks: desktopicon

[Run]
Filename: "{app}\FlowFI.exe"; Description: "{cm:LaunchProgram,FlowFI}"; Flags: nowait postinstall skipifsilent
