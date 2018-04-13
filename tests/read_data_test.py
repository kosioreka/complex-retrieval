from trec_car.read_data import *
import sys


if len(sys.argv)<4:
    print("usage ",sys.argv[0]," articlefile outlinefile paragraphfile")
    exit()

articles=sys.argv[1]
outlines=sys.argv[2]
paragraphs=sys.argv[3]


# to open either pages or outlines use iter_annotations

with open(articles, 'rb') as f:
    for p in iter_pages(f):
        print('\npagename:', p.page_name)
        print('\npageid:', p.page_id)
        print('\nmeta:', p.page_meta)

        # get one data structure with nested (heading, [children]) pairs
        headings = p.nested_headings()
        print(headings)

        if len(p.outline())>0:
            print('heading 1=', p.outline()[0].__str__())

            print('deep headings= ', [ (section.heading, len(children)) for (section, children) in p.deep_headings_list()])
            flat_headings = {}
            sections_list = p.flat_headings_list()
            for sectionpath in sections_list:
                #section path gives hierarchy of the sections as a list
                list1 = [" / ".join([section.heading for section in sectionpath])] # returns sectionpath[0] / sectionpath[1]...
                para_children = []
                for cb in sectionpath[-1].children:
                    if hasattr(cb, 'paragraph'):
                        para_children.append(' '.join(
                            [c.text if isinstance(c, ParaText) else c.anchor_text for c in cb.paragraph.bodies]))
                    elif hasattr(cb, 'body'):
                        para_children.append(' '.join(
                            [c.text if isinstance(c, ParaText) else c.anchor_text for c in cb.body.bodies]))
                section = [(p.page_name+" / ") + l for l in list1]
                flat_headings[section[0]] = ' '.join(para_children)
            print('flat headings1= ', [s for s in flat_headings])
            print('flat headings=  ' ,["/".join([section.heading for section in sectionpath]) for sectionpath in p.flat_headings_list()])
        break
# exit(0)

with open(outlines, 'rb') as f:
    for p in iter_outlines(f):
        print('\npagename:', p.page_name)

        # get one data structure with nested (heading, [children]) pairs
        headings = p.nested_headings()
        print(headings)
        print([h[0].heading for h in headings])

        if len(p.outline())>2:
            print('heading 1=', p.outline()[0])
        # print('deep headings= ',  p.deep_headings_list())
        deep = p.deep_headings_list()
        flat = p.flat_headings_list()
        if len(deep) != len(flat):
            a = deep
        print('deep headings= ', [h[0].heading for h in p.deep_headings_list()])


        # print('flat headings= ', p.flat_headings_list())
        print('flat headings= ', [h[0].heading for h in p.flat_headings_list()]) #h[0] is the main heading, h[1] if exists contains a child section
        break


with open(paragraphs, 'rb') as f:
    for p in iter_paragraphs(f):
        print('\n', p.para_id, ':')

        # Print just the text
        texts = [elem.text if isinstance(elem, ParaText)
                 else elem.anchor_text
                 for elem in p.bodies]
        print(' '.join(texts))

        # Print just the linked entities
        entities = [elem.page
                    for elem in p.bodies
                    if isinstance(elem, ParaLink)]
        print('entities: ', entities)

        # Print text interspersed with links as pairs (text, link)
        mixed = [(elem.anchor_text, elem.page) if isinstance(elem, ParaLink)
                 else (elem.text, None)
                 for elem in p.bodies]
        print('mixed: ', mixed)
        break
