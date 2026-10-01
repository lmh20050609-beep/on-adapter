import unittest
from collections import Counter
from iadapter.text_order import recent_entry

class TextOrderTests(unittest.TestCase):
    def entry(self):
        old='previous event with stable production';new='latest event with falling fuel prices';text='Facts. '+old+' Facts. '+new
        return dict(text=text,spans=[('5:fact',7,7+len(old)),('6:fact',len('Facts. '+old+' Facts. '),len(text))])
    def test_full_blocks_preserve_characters_and_source_spans(self):
        entry=self.entry();changed=recent_entry(entry);self.assertTrue(changed['text'].startswith('Facts. latest'));self.assertEqual(Counter(entry['text']),Counter(changed['text']))
        old={rid:entry['text'][a:b] for rid,a,b in entry['spans']};new={rid:changed['text'][a:b] for rid,a,b in changed['spans']};self.assertEqual(old,new)
    def test_preexisting_character_crop_is_not_replaced_with_more_text(self):
        entry=self.entry();left=16;clipped=dict(text=entry['text'][left:],spans=[(rid,max(0,a-left),b-left) for rid,a,b in entry['spans'] if b>left]);changed=recent_entry(clipped)
        self.assertEqual(Counter(changed['text']),Counter(clipped['text']));self.assertNotIn('previous event',changed['text'])
    def test_one_retained_row_and_empty_input_are_unchanged(self):
        for entry in [dict(text='',spans=[]),dict(text='one retained row',spans=[('7:fact',0,16)])]:
            self.assertEqual(recent_entry(entry)['text'],entry['text']);self.assertFalse(recent_entry(entry)['reordered'])
if __name__=='__main__':unittest.main()
