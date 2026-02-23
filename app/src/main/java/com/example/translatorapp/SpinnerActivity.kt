import android.app.Activity
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Spinner
import com.example.translatorapp.R

class SpinnerActivity : Activity(), AdapterView.OnItemSelectedListener {

    private lateinit var targetLanguageSpinner: Spinner
    private var targetLanguage: String? = null  // stores selected language

    public override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)  // use your layout file name

        // Find the Spinner by its ID
        targetLanguageSpinner = findViewById(R.id.targetLanguage)
        targetLanguageSpinner.onItemSelectedListener = this

        // Populate Spinner with full language names
        val languages = listOf("English", "Chinese", "Japanese", "Korean")
        val adapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, languages)
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        targetLanguageSpinner.adapter = adapter
    }

    // Called when the user selects an item
    override fun onItemSelected(parent: AdapterView<*>, view: View?, pos: Int, id: Long) {
        targetLanguage = parent.getItemAtPosition(pos) as String
        Log.d("SpinnerActivity", "Selected target language: $targetLanguage")
    }

    override fun onNothingSelected(parent: AdapterView<*>) {
        targetLanguage = null
    }
}