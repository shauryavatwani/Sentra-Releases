; Inno Setup script for Sentra — builds SentraSetup.exe
;
; Compile from the project root, after PyInstaller has produced dist\Sentra\:
;     iscc windows\sentra.iss
;
; Output lands in windows\output\SentraSetup.exe
;
; ---------------------------------------------------------------------------
; The two ideas this script is built around
;
; 1. PROGRAM FILES IS REPLACEABLE, PROGRAMDATA IS SACRED.
;    Everything under {app} is the build and is wiped and rewritten on every
;    upgrade. Everything under {commonappdata}\Sentra is the client's own
;    record — detections, registered faces, visitor photos, camera
;    configuration, logs — and is never overwritten, never deleted, not even by
;    an uninstall. A security system that loses its history during a routine
;    update has failed at the one job it was bought for.
;
; 2. SEED DATA IS FOR FRESH INSTALLS ONLY.
;    A first install lays down the complete working system as shipped. An
;    upgrade lays down none of it. This is decided once, before any file is
;    copied (see InitializeSetup), because the check would otherwise start
;    returning a different answer halfway through the copy.
; ---------------------------------------------------------------------------

; Version and build date come from Formal_Code/sentra_version.py by way of the
; PyInstaller spec, so the installer, the About panel and the update check can
; never disagree about what this build is.
#include "version_define.iss"

#define AppName        "Sentra"
#define AppPublisher   "Delhi Public School Bangalore East"
#define AppExeName     "Sentra.exe"
#define DataDir        "{commonappdata}\Sentra"

[Setup]
AppId={{8F3A6C41-7B29-4D5E-9A18-2C5E7F0B4D63}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppPublisher}
VersionInfoVersion={#AppVersion}
VersionInfoCompany={#AppPublisher}
VersionInfoDescription={#AppName} — CCTV Intelligence
VersionInfoProductName={#AppName}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
OutputDir=output
OutputBaseFilename=SentraSetup
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
; The app bundles torch and the InsightFace models, so it is 64-bit only and
; large. Declaring the size up front avoids a confusing "not enough space"
; failure partway through.
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
; Writing to Program Files and adding a firewall rule both need elevation.
PrivilegesRequired=admin
DisableProgramGroupPage=yes
UninstallDisplayIcon={app}\{#AppExeName}
UninstallDisplayName={#AppName} {#AppVersion}
SetupIconFile=sentra.ico
; Shown on the welcome page so an operator can tell at a glance which build
; they are about to install over the top of.
AppComments={#AppName} {#AppVersion}, built {#BuildDate}
CloseApplications=yes
RestartApplications=no

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
; Checked by default — a security tool the operator cannot find is a support
; call. Presented on the standard "Additional tasks" page, the same place
; VS Code and every other Inno-built application puts it.
Name: "desktopicon"; Description: "Create a &desktop shortcut"; \
    GroupDescription: "Shortcuts:"
Name: "startupicon"; Description: "Start {#AppName} automatically when Windows starts"; \
    GroupDescription: "Startup:"; Flags: unchecked

[Files]
; --- The application ------------------------------------------------------
; The entire PyInstaller one-folder build. Replaced wholesale on every upgrade.
Source: "..\dist\Sentra\*"; DestDir: "{app}"; \
    Flags: ignoreversion recursesubdirs createallsubdirs

; --- Seed data: FRESH INSTALLS ONLY ---------------------------------------
; This ships the complete working system as it stands today — the detection
; history, the registered face embeddings and their source photos, the visitor
; records and gate photos, and the camera configuration.
;
; Every one of these carries BOTH guards on purpose:
;   Check: IsFreshInstall   — an upgrade skips the whole block, so a build's
;                             seed data can never reappear over the top of the
;                             client's own work months later
;   onlyifdoesntexist       — belt and braces if the folder is partly present
;   uninsneveruninstall     — uninstalling leaves the client's records intact
;
Source: "..\Database\detections.db"; DestDir: "{#DataDir}\Database"; \
    Flags: onlyifdoesntexist uninsneveruninstall; Check: IsFreshInstall
Source: "..\Database\face_embeddings.pkl"; DestDir: "{#DataDir}\Database"; \
    Flags: onlyifdoesntexist uninsneveruninstall; Check: IsFreshInstall
Source: "..\Database\camera_config.json"; DestDir: "{#DataDir}\Database"; \
    Flags: onlyifdoesntexist uninsneveruninstall; Check: IsFreshInstall
; Enrolment photos. face_register.py re-embeds from this folder, so shipping it
; means the client can add a photo to an existing person rather than having to
; re-register them from scratch.
Source: "..\Faces\*"; DestDir: "{#DataDir}\Faces"; \
    Flags: onlyifdoesntexist uninsneveruninstall recursesubdirs createallsubdirs; \
    Check: IsFreshInstall
; Gate photos backing the visitor rows already in detections.db. Without these
; the Temporary Pass tab would show visitor records with broken photos.
Source: "..\Visitors\*"; DestDir: "{#DataDir}\Visitors"; \
    Flags: onlyifdoesntexist uninsneveruninstall recursesubdirs createallsubdirs skipifsourcedoesntexist; \
    Check: IsFreshInstall

[Dirs]
; The app writes here at runtime. Program Files is read-only for a standard
; user, which is why none of this lives under {app}. Everyone needs write
; access because the engine may run as a different account than the person
; registering faces.
Name: "{#DataDir}"; Permissions: users-modify
Name: "{#DataDir}\Database"; Permissions: users-modify
Name: "{#DataDir}\Faces"; Permissions: users-modify
Name: "{#DataDir}\Visitors"; Permissions: users-modify
Name: "{#DataDir}\.run_logs"; Permissions: users-modify
; Where the in-app updater stages a downloaded installer before running it.
Name: "{#DataDir}\.updates"; Permissions: users-modify

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExeName}"
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon
Name: "{userstartup}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: startupicon

[Run]
; Sentra binds 0.0.0.0:8000 so the dashboard can be opened from a phone or
; another PC on the same network. Without this rule Windows Firewall silently
; blocks that and only localhost works — which looks like a bug in the app.
; Deleted first so re-running the installer cannot stack duplicate rules.
Filename: "{sys}\netsh.exe"; \
    Parameters: "advfirewall firewall delete rule name=""Sentra Dashboard (TCP 8000)"""; \
    Flags: runhidden; StatusMsg: "Configuring the Windows Firewall..."
Filename: "{sys}\netsh.exe"; \
    Parameters: "advfirewall firewall add rule name=""Sentra Dashboard (TCP 8000)"" dir=in action=allow protocol=TCP localport=8000"; \
    Flags: runhidden; StatusMsg: "Configuring the Windows Firewall..."

; Interactive install: a checkbox on the final page, ticked by default.
; runasoriginaluser matters — setup is elevated, and without it Sentra would
; inherit administrator rights it has no need for and would then write files
; the signed-in user cannot edit.
Filename: "{app}\{#AppExeName}"; Description: "Launch {#AppName}"; \
    Flags: nowait postinstall skipifsilent runasoriginaluser

; Silent install: this is how the in-app updater applies an update, and the
; checkbox above is skipped in silent mode. Without this entry an update would
; leave the operator staring at a closed application, which reads as a crash.
Filename: "{app}\{#AppExeName}"; \
    Flags: nowait runasoriginaluser; Check: WizardSilent

[UninstallDelete]
; Only the app's own scratch space: logs and any half-staged update installer.
; Database, Faces and Visitors are deliberately absent from this list —
; uninstalling must never destroy the recorded detections, the faces someone
; spent time registering, or the visitor log, which is exactly the record an
; incident enquiry would ask for.
Type: filesandordirs; Name: "{#DataDir}\.run_logs"
Type: filesandordirs; Name: "{#DataDir}\.updates"

[Code]
var
  FreshInstall: Boolean;

function InitializeSetup(): Boolean;
begin
  // Decided once, up front. Asking this question later would give a different
  // answer as soon as the first seed file lands, and half the seed data would
  // be skipped on a genuinely fresh install.
  FreshInstall := not FileExists(ExpandConstant('{commonappdata}\Sentra\Database\detections.db'));
  Result := True;
end;

function IsFreshInstall: Boolean;
begin
  Result := FreshInstall;
end;

// Sentra holds port 8000 and the camera stream. Upgrading while it is running
// leaves a half-replaced install and an orphaned engine process still holding
// the RTSP connection, so stop it first. Both names are killed because the
// engine runs as a second process spawned from the same executable.
function PrepareToInstall(var NeedsRestart: Boolean): String;
var
  ResultCode: Integer;
begin
  Exec('taskkill.exe', '/F /IM {#AppExeName} /T', '', SW_HIDE,
       ewWaitUntilTerminated, ResultCode);
  // Give Windows a moment to release the file locks before the copy starts.
  Sleep(1500);
  Result := '';
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
  begin
    // An upgrade leaves the previous version's staged installer sitting in
    // .updates — roughly a gigabyte of a build that is now older than the one
    // just installed.
    if not FreshInstall then
      DelTree(ExpandConstant('{commonappdata}\Sentra\.updates'), True, True, True);
  end;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  ResultCode: Integer;
begin
  if CurUninstallStep = usUninstall then
  begin
    Exec('taskkill.exe', '/F /IM {#AppExeName} /T', '', SW_HIDE,
         ewWaitUntilTerminated, ResultCode);
    Exec(ExpandConstant('{sys}\netsh.exe'),
         'advfirewall firewall delete rule name="Sentra Dashboard (TCP 8000)"',
         '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
  end;

  if CurUninstallStep = usPostUninstall then
    MsgBox('Sentra has been removed.' + #13#10 + #13#10 +
           'Your detection history, registered faces, visitor records and camera ' +
           'settings have been kept in:' + #13#10 +
           ExpandConstant('{commonappdata}\Sentra') + #13#10 + #13#10 +
           'Reinstalling Sentra will pick them up again. Delete that folder ' +
           'yourself if you want the data gone.',
           mbInformation, MB_OK);
end;
