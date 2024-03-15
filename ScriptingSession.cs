using Grooper;
using Grooper.Core;
using Grooper.Services;

namespace GrooperCV
{
///<summary>Grooper SciptingSession object. 2023 by BIS (http://www.grooper.com).</summary>
///<remarks>
///Welcome to Grooper Scripting!  Here is some quick information for beginners:
///
///1) Member variables of this class will stay intact between Initialize() and Uninitialize().
///2) Please note that compiling this script from Visual Studio will not add/update the ObjectLibrary DLL in Grooper.  To add/update 
///   this DLL in Grooper and make the functionality in this project visible to Grooper, the Compile button must be 
///   used from the Script tab of the Grooper node in the Grooper Web Client.
///3) Object Library scripts can be used to customize the way that Grooper works.  This can be done by extending a 
///   Grooper class through inheritance or by creating methods that can be called from within Grooper.
///4) Data Model and Batch Process Scripts operate primarily by responding to events on the object to which they are
///   attached.  Viewing a list of events your script can handle in Visual Studio: select ObjectLibrary from the objects dropdown 
///   above, and then view the right-most dropdown for a list of events.
///5) You can debug with the GrooperSDK.  Find the GrooperSDK in Visual Studio by going to Extensions..Manage Extensions..Online and searching for 'Grooper'.  
///   In addition to debugging, the GrooperSDK exposes commands such as Save, Save and Compile, and Get Latest.
///6) More information can be found using Grooper Help (open Grooper and click F1) or the Grooper Wiki: https://wiki.grooper.com/.
///</remarks>
  public class ScriptingSession : ScriptObject
  {
    private ObjectLibrary ObjectLibrary;
    private GrooperRoot Root;

    /// <inheritdoc/>
    public override bool Initialize(GrooperNode Item)
    {
      ObjectLibrary = (ObjectLibrary)Item;
      Root = Item.Root;
      return true;
    }

    /// <inheritdoc/>
    public override bool Uninitialize() => true;
  }
}
