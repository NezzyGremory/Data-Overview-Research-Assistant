#define AppName "Data Overview & Research Assistant"
#define AppVersion "2.0.0"
#define AppPublisher "Metriks Studio"
#define AppExeName "DORA.exe"

[Setup]
AppId={{6A358E27-5C04-49D8-B3D7-7CC448928AE1}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
VersionInfoVersion={#AppVersion}
VersionInfoProductVersion={#AppVersion}
VersionInfoCompany={#AppPublisher}
VersionInfoDescription={#AppName}
VersionInfoProductName={#AppName}
DefaultDirName={localappdata}\Programs\DORA
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=..\hasil_compile
OutputBaseFilename=Data-Overview-Research-Assistant-v2.0.0-Setup
SetupIconFile=..\app\ui\assets\dataset_research.ico
UninstallDisplayIcon={app}\{#AppExeName}
ArchitecturesInstallIn64BitMode=x64compatible
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
CloseApplications=yes
RestartApplications=no
Uninstallable=yes
CreateAppDir=yes

[Files]
Source: "..\hasil_compile\Data-Overview-Research-Assistant-v2.0.0-runtimefix\payload\DORA\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\{#AppExeName}"; WorkingDir: "{app}"
Name: "{autodesktop}\DORA"; Filename: "{app}\{#AppExeName}"; WorkingDir: "{app}"

[Run]
Filename: "{app}\{#AppExeName}"; Description: "Launch {#AppName}"; Flags: postinstall nowait skipifsilent
